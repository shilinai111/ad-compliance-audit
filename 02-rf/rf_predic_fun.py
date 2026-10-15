import jieba
import pickle
from config import Config
import time

config =Config()

#加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

#定义推理函数用于预测
def predict_fun(data):
    words = ' '.join(jieba.lcut(data['text'])[:20])
    #计算tf-idf向量矩阵
    #输入必须是可迭代对象
    features = tfidf_vectorizer.transform([words])

    #模型预测
    y_pred = rf_model.predict(features)

    # print(y_pred)

    #分类返回文字
    with open(config.class_path, 'r',encoding='utf-8') as f:
        class_list = [line.strip() for line in f if line.strip()]

    pred_class  = class_list[y_pred[0]]

    data['pred_class']=pred_class
    return data
if __name__ == '__main__':
    data = {'text':'阿俊喜欢广亮'}
    output = predict_fun(data)
    print(output)

    start_time = time.time()
    for i in range(100):
        result = predict_fun(data)
    end_time = time.time()
    print(f'平均推理时长:{(end_time-start_time)/100}s')