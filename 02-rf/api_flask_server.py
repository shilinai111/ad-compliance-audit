from flask import Flask, jsonify, request
#request 用于向服务器发送网络请求 或者 服务器段用于接受数据
#jsonify讲有python数据转化为json格式才能在服务器段和客户端之间传送
from rf_predic_fun import *
from config import *

config = Config()
app = Flask(__name__)
# —— 跨域支持（CORS）：供 06-web/index.html 这类本地静态页面直接调用 ——
# 浏览器从 file:// 或其它端口访问本机服务属于跨域，会先发 OPTIONS 预检请求。
# 这里放行 OPTIONS 并回写 CORS 头，不引入任何新依赖。
@app.after_request
def _add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    # Chrome 的「私有网络访问」预检会额外要求这个头（与 04_Bett 保持一致）
    resp.headers['Access-Control-Allow-Private-Network'] = 'true'
    return resp


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():   # 绑定一个视图函数， 当别人利用 GET方法访问这个地址url，应该做些什么。
    # 跨域预检请求：直接回 204，不必进模型
    if request.method == 'OPTIONS':
        return ('', 204)

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


