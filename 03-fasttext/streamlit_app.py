import streamlit as st
import requests
import time
from config import Config

config = Config()

st.title('广告文案合规审核系统（fastText）')
st.caption('类别：不违规 / 黑产广告 / 欺诈 / 博彩 / 谩骂引战 / 低俗色情')

text = st.text_input('请输入待审核的文案:')

url = 'http://'+config.api_host+':'+config.api_port+'/predict'

if st.button('fasttext预测'):
    try:
        start = time.time()

        #获得的文本封装为字典
        data = {'text':text}

        #使用requests发送json
        result = requests.post(url,json=data)

        print('预测结果',result.json())

        total_time = (time.time() - start)*1000.

        #在页面上展示推理的类别和时间
        st.success('预测的类别:'+ result.json()['pred_class']+'|推理的时时长'+str(round(total_time,2)) +'ms')


    except Exception as e:
        st.error('错误！请联系管理员胡广亮')
        print('error',e)
