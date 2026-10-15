'''
演示
    flask框架入门操作，创造一个简单的API服务
需要掌握：
    安装flask, pip install flask
    app = Flask(__name__)
    @app.route('/', methods=['GET','POST'])
'''

from flask import Flask, render_template

# 指定静态资源目录和模板目录
app = Flask(__name__, static_folder='imgs', template_folder='templates')


@app.route('/', methods=['GET'])
def home():
    # 炫酷欢迎页
    return render_template('welcome.html')


@app.route('/start', methods=['GET'])
def start():
    # 点击开始后显示图片页
    return render_template('show_image.html', image_name='img.jpg')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
