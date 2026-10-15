'''
演示
    flask框架入门操作，创造一个简单的API服务

    将本地笔记本电脑当做服务器 ，运行运行起来，并指定谁可以用当前的服务器

需要掌握：
    安装flask, pip install flask
    app = Flask(__name__)
    @app.route('/', methods=['GET','POST'])
'''

# Flask 类可以创建 web 应用
# 1  管理路由（URL映射） 2 处理请求和响应  3管理配置
# send_file 发送文件的函数         render_template 渲染HTML模板的函数
from flask import Flask, send_file, render_template

# 1.创建 Flask 实例
app = Flask(__name__)   # __name__表示当前模块名

# 2.创建预测接口 (  创建路由 + 处理函数)
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
# URL: http://服务器地址：端口号/
@app.route('/abc', methods=['GET'])
def predict1():   # 绑定一个预测函数
    #return "hello world"

    # 返回图片
    # mimetype：明确告诉浏览器：这是一个JPEG格式的图片，让浏览器决定如何下载和展示
    return send_file('imgs/img.jpg', mimetype='image/jpeg')


    # 渲染首页模板
    #读取index.html文件，把它转换成完整的HTML网页，然后返回给浏览器显示。
    #return render_template('index.html')



# 启动 web API 服务
if __name__ == '__main__':
    '''注意，现在你的笔记本就是一个服务器，   host.指定服务器监听的IP地址'''
    #  '127.0.0.1'是  本机地址，且只允许本机访问   端口号5001
    app.run(host='127.0.0.1', port=5001, debug=True)

    ''' 如果 URL 路径被修改  @app.route('/abc', methods=['GET'])
        则你的访问网址依然是 ： app.run(host='127.0.0.1', port=5001, debug=True)
        但是打开是的页面网址需要追加子路径  127.0.0.1/abc
    '''
    # 2.允许同一局域网设备访问你的电脑中的服务器，则告知你本机的 IP
    # app.run(host='192.168.1.4', port=5000, debug=True)
    # 3.允许所有设备访问你的电脑的服务器，包括局域网和外网
    # app.run(host='0.0.0.0', port=5000, debug=True)
