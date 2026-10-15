'''
对官网的接口代码进行测试
'''

import os  #操作系统接口模块，用于读取系统环境变量
# openai库中调用 OpenAI 的客户端类  用于创建客户端实例
from openai import OpenAI    # pip install openai  -i

# 加载环境变量的函数
from dotenv import load_dotenv


# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

load_dotenv('.env')
# 将.env文件中的环境变量（API key）添加到系统的环境变量中


def call_deepseek_api(user_prompt, system_prompt='You are a helpful assistent'):

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'), # 从系统环境变量中读取具体的API key
        base_url="https://api.deepseek.com")   # 请求发送的地址

    response = client.chat.completions.create(
        model="deepseek-flash",                # 模型选型 （看需求）
        messages=[
            {"role": "system", "content": system_prompt},  #系统提示词，辅助提示词
            {"role": "user", "content": user_prompt},  # 用户提示词  新闻标题
        ],
        stream=False,            # 流式输出
        reasoning_effort="high", # 推理强度
        extra_body={"thinking": {"type": "enabled"}}
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    result= call_deepseek_api('亚运会中国奖牌总数第一。','请用文言文进行回答，体现诗情画意')
    print(result)