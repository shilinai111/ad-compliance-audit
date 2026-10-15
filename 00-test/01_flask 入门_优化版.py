'''
演示
    flask框架入门操作，创造一个简单的API服务，接受请求并给与响应

    类比：  医院（API服务）先上线，   病员（客户端）才能访问

    医院（路由器）需要公开具体地址（服务器IP）、
    并公示每位医生的就诊通道，急诊 门诊，vip（端口号）、     哪个科室（路径）

    同时医院需要对医生进行培训，针对病员（客户端）的不同的诉求（http请求方式），
    比如问诊，复诊，开药，医闹（GET  POST  PUT  DELETE）, 医生需要如何应对（视图函数）。
    最后将处理的结果反馈给病员（客户端）

    # HTTP请求方法：GET（获取资源）, POST（提交数据）, PUT（更新资源）, DELETE（删除资源）


'''

# Flask 类可以创建 web 应用
# 1  管理路由（URL映射） 2 处理请求和响应  3管理配置
# send_file 发送文件的函数         render_template 渲染HTML模板的函数
from flask import Flask, send_file, render_template


# 1.创建 Flask 实例
app = Flask(__name__)   # __name__表示当前模块名



# 2.创建预测接口 (  创建路由 + 处理函数/视图函数，即服务器需要做什么事情)

# 创建路由，并指定路径
# 路由就是一张"地址-功能"的映射表，告诉Web服务器：当用户访问某个URL时，应该执行什么代码。
# 完整的URL = 协议 + IP地址 + 端口 + 路径  比如： http://159.226.71.206:5000/abc
#  类比：  著名的医生 在哪个地址的大楼的哪个房间（科室），
@app.route('/abc', methods=['GET'])
def predict1():   # 绑定一个视图函数， 当别人利用 GET方法访问这个地址url，应该做些什么。
    return "hello world"




@app.route('/def', methods=['GET'])
def predict2():  # 绑定一个视图函数， 但名字不能一样

    # 返回图片
    # mimetype：明确告诉浏览器：这是一个JPEG格式的图片，让浏览器决定如何下载和展示
    return send_file('imgs/img2.jpg', mimetype='image/jpeg')




'''
Flask 的路由装饰器，
methods=['GET'] ：指定该路由只接受HTTP的GET请求方法
     '/' ： 代表网站的首页或基础URL
            服务器上这个Flask应用的首页/入口
     GET是最常见的HTTP方法，用于从服务器获取数据
     浏览器地址栏直接访问、点击链接都属于GET请求
     
     其他方法： POST(服务器向客户端提交数据)
              PUT(更新服务器资源)   DELETE(删除服务器资源)
'''




# 启动 web API 服务
if __name__ == '__main__':
    '''注意，现在你的笔记本就是一个服务器，   host.指定服务器监听的IP地址'''
    #  '127.0.0.1'是  本机地址，且只允许本机访问   端口号5001
    app.run(host='127.0.0.1', port=5001, debug=True)

    ''' 如果 URL 路径被修改  @app.route('/abc', methods=['GET'])
        则你的访问网址依然是 ： app.run(host='127.0.0.1', port=5001, debug=True)
        但是打开是的页面网址需要追加子路径  127.0.0.1/abc  就能显示图片
    '''

    # 具体IP地址
    #服务器有多个网卡时的特殊需求   只监听服务器的特定网卡 输入该网卡的IP
    # app.run(host='159.226.71.202', port=5000, debug=True)

    #让服务器监听所有网络接口，这样无论是通过内网IP、公网IP还是localhost都能访问
    # app.run(host='0.0.0.0', port=5000, debug=True)
