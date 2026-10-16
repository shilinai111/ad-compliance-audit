'''
演示
   用过flask工具，构建  路由+视图函数 从而部署一个服务
'''

from flask import Flask, request, jsonify
# request  用于向服务器发送网络请求  或者 服务端用于接收数据
# jsonify 将有python数据转化为Json格式才能在服务端和客户端之间传输
'''{
  "name": "张三",
  "age": 25,
  "is_student": false,
  "score": 95.5,
  "hobbies": ["读书", "游泳", "编程"]
}'''

from deepseek_predict_fun import predict_fun
#from config import Config


#config=Config()

# 1.创建 Flask 实例
app = Flask(__name__)   # __name__表示当前模块名



# 2.创建预测接口 (  创建路由 + 处理函数/视图函数，即服务器需要做什么事情)

# 创建路由，并指定路径
# 路由就是一张"地址-功能"的映射表，告诉Web服务器：当用户访问某个URL时，应该执行什么代码。
# 完整的URL = 协议 + IP地址 + 端口 + 路径  比如： http://159.226.71.206:5000/abc
#  类比：  著名的医生 在哪个地址的大楼的哪个房间（科室），
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
def predict1():   # 绑定一个视图函数， 当别人利用 GET方法访问这个地址url，应该做些什么。
    # 跨域预检请求：直接回 204，不必调大模型
    if request.method == 'OPTIONS':
        return ('', 204)

    # 收到的数据是json格式，需要先转化为 python对象
    data = request.get_json()

    # 将转化好的数据 送入随机森林模型, 获得模型预测的结果（字典格式）
    result = predict_fun(data)

    return jsonify(result)

# 启动 web API 服务
if __name__ == '__main__':
    '''注意，现在你的笔记本就是一个服务器，   host.指定服务器监听的IP地址'''
    #  '127.0.0.1'是  本机地址，且只允许本机访问   端口号5001
    app.run(host='127.0.0.1', port=5002, debug=True)


























