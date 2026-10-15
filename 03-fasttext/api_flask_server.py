from flask import Flask, jsonify, request
#request 用于向服务器发送网络请求 或者 服务器段用于接受数据
#jsonify讲有python数据转化为json格式才能在服务器段和客户端之间传送
from fasttext_predict_fun import predict_fun
from config import Config

config = Config()
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():   # 绑定一个视图函数， 当别人利用 GET方法访问这个地址url，应该做些什么。
    #收到的json数据需要转换为python对象
    data = request.get_json()

    #讲转换好的数据送入随机森林 获得模型预测结果
    result  =predict_fun(data)
    # print(f"客户端发来的json数据: {data}")
    # if data and "text" in data:
    #     print(f"用户输入文本：{data['text']}")
    return jsonify(result)

if __name__ == '__main__':

    app.run(host = config.api_host,port = config.api_port,debug = True)


