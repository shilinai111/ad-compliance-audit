'''
  进行提示词工程，让大模型为你生成合理的系统提示词，用于自己文本数据的推理

  业务场景（与原 THUCNews 新闻分类项目的差异）：
    本模块现在服务于「广告素材 / 投放文案合规审核」场景，
    类别与 01-data/class.txt 严格保持一致（6 类中文标签），
    从而可以和 04_Bett 的微调 BERT 做同一份测试集上的横向对比。
'''




import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
1. **唯一性**: 对每个输入文本，只能输出一个最匹配的类别名称。
2. **格式**: 仅输出类别名称本身（中文），严禁输出任何解释、序号、标点符号或额外文本。
3. **范围**: 输出必须严格限制在以下六个之一：
   `不违规`、`黑产广告`、`欺诈`、`博彩`、`谩骂引战`、`低俗色情`。
4. **拒判机制**: 若文本过短（少于 4 个字）、语义完全无法判断，或你确实无法在六类中做出选择，
   输出 `需人工复核`。宁可转人工，也不要瞎猜。

# Few-Shot Examples

Input: 他吃完饭离开了餐厅。
Output: 不违规

Input: 可有偿助力服务，助力一次0.5，可助力五次到六次，需要先加我
Output: 黑产广告

Input: 店铺好评，销量提升联系我/我最美
Output: 黑产广告

Input: 不用出门一天轻轻松松二三百，不收费需要滴滴
Output: 欺诈

Input: 来海外直播平台，月入十万不香吗？在这扭半天也没啥流量
Output: 欺诈

Input: n33qg.topn81ug.top这里可以打麻将
Output: 博彩

Input: 浏览器搜索yhh09.top
Output: 博彩

Input: 找骂的玩意，脑can
Output: 谩骂引战

Input: 日本狗菜成这样也敢叫，真不愧是日本狗喔
Output: 谩骂引战

Input: 济南狗奴隶求主人调教！有空房可以给主人住！
Output: 低俗色情

Input: 91主播哈哈哈哈，他是卖的吧
Output: 低俗色情

Input: 好
Output: 需人工复核

# Workflow
1. 接收待审核的文案文本。
2. 扫描强特征：陌生域名 / 短链 → 博彩；钱+轻松赚 → 欺诈；刷量涨粉 → 黑产广告。
3. 无强特征时再判断是否存在辱骂或色情语义。
4. 以上都不成立 → 不违规；仍无法判断 → 需人工复核。
5. 只输出类别名称。"""




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

# 合法输出集合：与 01-data/class.txt 的 6 个类别保持一致，另加兜底项「需人工复核」
VALID_CLASSES = set(["不违规", "黑产广告", "欺诈", "博彩", "谩骂引战", "低俗色情"])
REVIEW_CLASS = "需人工复核"


def predict_fun(data):
    text = data['text']

    pred_class = call_deepseek_api(text, SYSTEM_PROMPT).strip()

    # 大模型偶尔会夹带标点、前后缀或输出范围外的类别，这里做一次兜底归一
    if pred_class not in VALID_CLASSES:
        hit = [c for c in VALID_CLASSES if c in pred_class]
        pred_class = hit[0] if len(hit) == 1 else REVIEW_CLASS

    data['pred_class'] = pred_class
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







