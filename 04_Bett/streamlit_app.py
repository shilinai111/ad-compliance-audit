"""
实现一个streamlit框架的前端页面 ，调用web服务
（优点：服务器和客户端之间建立稳定持久的双向通信通道，不会请求+响应一次就关闭。）

# 另一个常用的叫  Gradio 以后讲langchain的时候再讲   另一个轻量化的框架

仅仅需要掌握：
1 生成一个前端页面，可以输入文本，得到服务器响应的结果
2streamlit库: pip install streamlit
3 启动  streamlit run app.py

代码要求能看懂，不要求默写。因为AI可以生成。


主要步骤
0.创建配置对象
1.配置streamlit页面


2.向flask后端发送请求
2.1  构造网址url


2.2  设置预测按钮的执行逻辑
（做什么操作，调用后端，将返回的结果返回给前端）
"""

# 导包
import streamlit as st  # 用于生成前端页面的工具
import requests         # 用于向网站或 API 发送网络请求 或者 获取响应数据。
import time
from config import Config


# 创建配置对线
config=Config()

# 配置页面
# 设置页面标题
st.title('广告素材合规审核系统')
st.caption('输入待投放的广告文案 / 素材文本，自动判断是否违规')

# 获取在页面中输入的文本
text = st.text_input('请输入待审核文本：')

# 定义接口的地址 URL
url = 'http://' + config.api_host+':'+config.api_port+'/predict'
#print(url)


# 设置预测的按钮
if st.button('开始审核'):

     # 尝试运行
     try:
            # 记录开始时间
            start_time = time.time()

            # 将已获得的文本数据，封装成 字典
            data = {'text': text}

            # 使用requests 向服务端发送请求
            result = requests.post(url, json=data)

            res = result.json()   # 将返回的结果，转回python对象（字典）
            print('预测结果，', res)

            total_time = (time.time() - start_time) *1000.

            pred_class = res['pred_class']
            confidence = res.get('confidence', 0)
            need_review = res.get('need_review', False)

            # 在页面上展示审核结论、置信度和推理时长
            if need_review:
                # 模型没把握，走人工通道，用警告样式提醒
                st.warning(
                    f'判定结果：{pred_class}（模型倾向：{res.get("raw_class", "-")}）\n\n'
                    f'置信度仅 {confidence}，低于阈值 {config.conf_threshold}，建议转人工复核'
                )
            else:
                st.success( f'判定结果：{pred_class}' )

            st.info(f'置信度：{confidence}　|　推理耗时：{round(total_time, 2)} ms')


     except Exception as e:
         st.error('出问题了，请联系管理员')
         print('error',e)

     # '''
     # 完整流程
     #
     # 先开启服务器， 再调用   streamlit run streamlit_app.py
     #
     # 梳理逻辑   但是脚本不用记，都是标准流程，可以用AI生成
     # 1  开启服务端， api_flask_server, 定义路由，接受请求
     # 2  用streamlit开启客户端，输入文字，向指定的url 发送post请求,  发送的数据格式为json
     #     r = requests.post(url, json=data)
     # 3  服务器接收到了POST请求，执行对应的视图函数 predict
     #         @app.route('/predict', methods=['POST'])
     #
     #         执行 def predict():
     #                 内部调用; result = predict_fun(data)
     #                 # 看一下predict_fun是啥？ rf_predict_fun.py 30行
     #
     # 4   视图函数predict将预测的类别信息序列化后，返回给客户端
     #      return jsonify(result)
     #
     #
     # 5   r.json()['pred_class'] 将返回的json字串转为（反序列化）字典，获得 标签 的类别
     #     st.success("预测结果：" + r.json()['pred_class'])
     #     并显示在前端页面。
     # '''
























