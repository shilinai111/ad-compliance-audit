import requests
from config import *
config = Config()

url = 'http://'+config.api_host+':'+config.api_port+'/predict'
# print(url)

try:
    text = input('请输入文本内容：')

    data = {'text':text}

    result = requests.post(url, json=data)
    print('预测结果:',result.json())


except Exception as e:
    print('系统错误，请联系管理员',e)