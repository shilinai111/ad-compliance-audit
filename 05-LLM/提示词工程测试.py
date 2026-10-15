'''
  提示词工程的试验场：单独调试 / 对比 SYSTEM_PROMPT 的效果。

  正式提示词统一放在 deepseek_predict_fun.py 中维护，这里直接 import 复用，
  避免两处提示词不一致。改完提示词先跑这个文件看效果，再回去跑推理。
'''




import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# 复用正式提示词，保证「调试用的提示词」和「线上推理用的提示词」永远是同一份
from deepseek_predict_fun import SYSTEM_PROMPT




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

if __name__ == '__main__':
    # 覆盖 6 个类别 + 故意留 2 个易混淆样本（域名类、赚钱话术类）观察模型是否判对
    test_samples = {
        '不违规':   '他吃完饭离开了餐厅。',
        '黑产广告': '店铺好评，销量提升联系我/我最美',
        '欺诈':     '加我好友，我带你。我手上有个非常好的项目',
        '博彩':     '浏览器搜索yhh09.top',
        '谩骂引战': '装逼的，老母猪人都不敢见，照片都不敢发，玩不起别来装',
        '低俗色情': '手好看，可以帮我🦌吗',
        '需人工复核': '好',
    }
    for expect, text in test_samples.items():
        pred = call_deepseek_api(text, SYSTEM_PROMPT).strip()
        flag = 'OK ' if pred == expect else 'MISS'
        print(f"[{flag}] 期望={expect:<6s} 实际={pred:<8s} 文本={text}")







