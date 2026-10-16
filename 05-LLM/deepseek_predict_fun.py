'''
  进行提示词工程，让大模型为你生成合理的系统提示词，用于自己文本数据的推理

  业务场景（与原 THUCNews 新闻分类项目的差异）：
    本模块现在服务于「广告素材 / 投放文案合规审核」场景，
    类别与 01-data/class.txt 严格保持一致（6 类中文标签），
    从而可以和 04_Bett 的微调 BERT 做同一份测试集上的横向对比。
'''




import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 类别顺序必须与 01-data/class.txt 一致（probs 数组按这个顺序排列）
CLASS_LIST = ["不违规", "黑产广告", "欺诈", "博彩", "谩骂引战", "低俗色情"]
REVIEW_CLASS = "需人工复核"

# 置信度阈值：与其它三条路线同名同义，低于该值转人工复核（设为 0 关闭）。
# 【重要】这里的"置信度"来自**模型自报的分数，未经校准**，与 BERT 的 softmax、
# 随机森林的投票比例都不是一个东西，数值不可横向比较，阈值必须分别标定。
CONF_THRESHOLD = 0.60


SYSTEM_PROMPT="""
你是一个专业的「广告素材 / 投放文案合规审核」助手。你的任务是审核一段即将投放的文案或素材文本，
判断它是否存在违规风险，并精准归类到预定义的 6 个类别之一。

# Category Definitions
请严格按照以下定义和索引映射关系进行分类：

- **不违规 (索引 0)**: 正常的日常表达、知识分享、生活记录、普通商业宣传。
  不包含任何色情、赌博、诈骗、引流推广或人身攻击内容。
- **黑产广告 (索引 1)**: 灰黑产引流与推广变现行为，包括但不限于：
  刷单刷量、代发视频、店铺好评/销量造假、有偿助力、涨粉引流、卖号、私域引流话术。
- **欺诈 (索引 2)**: 以骗取钱财或个人信息为目的的诈骗话术，包括但不限于：
  兼职刷单返利、"轻松日赚几百"、高额回报投资诱饵、冒充客服/公检法、钓鱼链接、诱导转账汇款。
- **博彩 (索引 3)**: 赌博及色情引流网站推广，典型特征是文本中夹带陌生域名、短链接、
  "注册地址/官网/备用网址"，或出现打麻将、捕鱼、彩票、赌场等赌博玩法描述。
- **谩骂引战 (索引 4)**: 辱骂、人身攻击、地域/性别/国家歧视、恶意挑衅、煽动对立与网络暴力。
- **低俗色情 (索引 5)**: 色情交易招嫖、露骨性描写、性暗示、色情资源引流、涉黄主播/平台推广。

# Disambiguation Rules（易混淆场景的判断优先级）
1. **同时带陌生域名 + 赌博玩法** → 优先判 `博彩`（域名是强特征）。
2. **"轻松赚钱/日入过万" 类话术**：若只是诱导你做任务赚钱 → `欺诈`；
   若是出售"助力/刷单/代发"服务本身 → `黑产广告`。
3. **骂人内容中含色情词汇**：以攻击为目的 → `谩骂引战`；以色情招揽为目的 → `低俗色情`。
4. **普通商品促销**（打折、优惠券、新品上市）无灰产特征 → `不违规`，不要过度敏感。

# Constraints
1. **唯一性**: 对每个输入文本，只能给出一个最匹配的类别（放在 JSON 的 class 字段）。
2. **格式**: 只输出一个 JSON 对象，严禁解释、序号、markdown 代码块或任何额外文本。
3. **范围**: class 必须严格限制在以下六个之一：
   `不违规`、`黑产广告`、`欺诈`、`博彩`、`谩骂引战`、`低俗色情`。
4. **拒判机制**: 若文本过短（少于 4 个字）、语义完全无法判断，或你确实无法在六类中做出选择，
   class 填 `需人工复核`（此时 scores 仍要给出你倾向的分布）。宁可转人工，也不要瞎猜。

# Few-Shot Examples（只示范「该判哪一类」，最终输出格式统一见文末 Output Format）

Input: 他吃完饭离开了餐厅。
类别: 不违规

Input: 可有偿助力服务，助力一次0.5，可助力五次到六次，需要先加我
类别: 黑产广告

Input: 店铺好评，销量提升联系我/我最美
类别: 黑产广告

Input: 不用出门一天轻轻松松二三百，不收费需要滴滴
类别: 欺诈

Input: 来海外直播平台，月入十万不香吗？在这扭半天也没啥流量
类别: 欺诈

Input: n33qg.topn81ug.top这里可以打麻将
类别: 博彩

Input: 浏览器搜索yhh09.top
类别: 博彩

Input: 找骂的玩意，脑can
类别: 谩骂引战

Input: 日本狗菜成这样也敢叫，真不愧是日本狗喔
类别: 谩骂引战

Input: 济南狗奴隶求主人调教！有空房可以给主人住！
类别: 低俗色情

Input: 91主播哈哈哈哈，他是卖的吧
类别: 低俗色情

Input: 好
类别: 需人工复核

# Workflow
1. 接收待审核的文案文本。
2. 扫描强特征：陌生域名 / 短链 → 博彩；钱+轻松赚 → 欺诈；刷量涨粉 → 黑产广告。
3. 无强特征时再判断是否存在辱骂或色情语义。
4. 以上都不成立 → 不违规；仍无法判断 → 需人工复核。
5. 按下面的 Output Format 输出 JSON。

# Output Format（必须严格遵守）
只输出一个 JSON 对象，形如：
{"class":"欺诈","scores":{"不违规":0.02,"黑产广告":0.10,"欺诈":0.80,"博彩":0.03,"谩骂引战":0.02,"低俗色情":0.03}}

scores 规则：
- 必须包含全部六个类别，值为 0~1 的数；**不必和为 1**，按相对大小给即可（系统会自行归一化）
- class 必须等于 scores 里最大的那个类别；若拒判则填 `需人工复核`
- 分数标定参考：强特征明确时最高分 0.85~0.98；边界/模糊样本 0.40~0.65；完全无法判断时六类分数都低且接近
"""




def call_deepseek_api(user_prompt, system_prompt='You are a helpful assistent'):

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'), # 从系统环境变量中读取具体的API key
        base_url="https://api.deepseek.com")   # 请求发送的地址

    response = client.chat.completions.create(
        model="deepseek-flash",                # 模型选型 （看需求）
        messages=[
            {"role": "system", "content": system_prompt},  #系统提示词，辅助提示词
            {"role": "user", "content": user_prompt},  # 用户提示词  待审核的广告文案
        ],
        stream=False,            # 流式输出
        reasoning_effort="high", # 推理强度
        extra_body={"thinking": {"type": "enabled"}}
    )

    return response.choices[0].message.content

# 合法输出集合：与 01-data/class.txt 的 6 个类别保持一致（REVIEW_CLASS 已在文件顶部定义）
VALID_CLASSES = set(CLASS_LIST)


def _extract_json(raw):
    """尽力从模型输出里取出那个 JSON 对象：容忍 ```json 代码围栏与前后多余文字。"""
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:  # noqa: BLE001
            pass
    return None


def _normalize_class(value):
    """把模型给出的类别文本归一到六类之一；无法归一则返回 None。"""
    if not isinstance(value, str):
        return None
    s = value.strip().strip('"\'`。，,')
    if s in VALID_CLASSES:
        return s
    if s == REVIEW_CLASS:
        return REVIEW_CLASS
    hit = [c for c in CLASS_LIST if c in s]
    return hit[0] if len(hit) == 1 else None


def _extract_probs(payload):
    """从 JSON 里取出六类分数并归一化为概率；缺项或非法值返回 None。"""
    if not isinstance(payload, dict):
        return None
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None
    values = []
    for name in CLASS_LIST:
        try:
            values.append(max(0.0, float(scores.get(name))))
        except (TypeError, ValueError):
            return None
    total = sum(values)
    return [v / total for v in values] if total > 0 else None


def predict_fun(data):
    text = data['text']
    raw = call_deepseek_api(text, SYSTEM_PROMPT)

    payload = _extract_json(raw)
    probs = _extract_probs(payload)

    model_class = None
    if isinstance(payload, dict):
        model_class = _normalize_class(payload.get("class") or payload.get("pred_class"))
    if model_class is None:
        model_class = _normalize_class(raw)      # JSON 解析失败时退回纯文本解析

    # 有分数时以分数最大项为准，保证「类别」与「置信度」严格自洽
    leaning, conf = None, None
    if probs is not None:
        top = max(range(len(probs)), key=lambda i: probs[i])
        leaning = CLASS_LIST[top]
        conf = probs[top]

    base = leaning or model_class                  # 模型真正的判断（可能为 None）

    reject_by_conf = (conf is not None and CONF_THRESHOLD > 0 and conf < CONF_THRESHOLD)
    reject_by_model = (model_class == REVIEW_CLASS)
    need_review = reject_by_conf or reject_by_model or (base is None)

    data['pred_class'] = REVIEW_CLASS if need_review else base
    data['raw_class'] = base if base else REVIEW_CLASS
    data['confidence'] = round(conf, 4) if conf is not None else None
    data['need_review'] = need_review
    if probs is not None:
        data['probs'] = [round(p, 4) for p in probs]
    return data


if __name__ == '__main__':
    # 覆盖 6 个类别 + 1 个拒判场景，验证提示词是否稳定
    samples = [
        "他吃完饭离开了餐厅。",
        "可有偿助力服务，助力一次0.5，可助力五次到六次，需要先加我",
        "不用出门一天轻轻松松二三百，不收费需要滴滴",
        "n33qg.topn81ug.top这里可以打麻将",
        "找骂的玩意，脑can",
        "济南狗奴隶求主人调教！有空房可以给主人住！",
        "好",
    ]
    for s in samples:
        result = predict_fun({'text': s})
        print(f"{ result['pred_class']:<6s} <- {s}")







