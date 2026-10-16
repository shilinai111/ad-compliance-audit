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

    # 取六类概率：随机森林的"概率"就是 200 棵树的投票比例。
    # 只有拿到概率才能做「置信度拒判」和前端概率条 —— 之前这里只调 predict()，概率被丢掉了。
    # 另外 predict() 内部也是先算 proba，两个都调会白跑一遍，所以这里只用 predict_proba。
    proba = rf_model.predict_proba(features)[0]

    #分类返回文字
    with open(config.class_path, 'r',encoding='utf-8') as f:
        class_list = [line.strip() for line in f if line.strip()]

    # 概率按 rf_model.classes_ 的顺序给出，用类别索引放回固定槽位，不依赖顺序假设
    probs = [0.0] * len(class_list)
    for cls, p in zip(rf_model.classes_, proba):
        idx = int(cls)
        if 0 <= idx < len(probs):
            probs[idx] = float(p)

    # 以概率最大的类别为准（比用 y_pred 更能保证与 confidence 严格对应）
    top = int(proba.argmax())
    pred_class = class_list[int(rf_model.classes_[top])]
    conf = float(proba[top])

    # 与 BERT 路线保持一致的拒判语义：低于阈值时 pred_class 换成「需人工复核」，
    # 原判保留在 raw_class 供人工参考；阈值与 BERT 同名（config.conf_threshold）
    threshold = getattr(config, 'conf_threshold', 0.0)
    need_review = threshold > 0 and conf < threshold

    data['pred_class'] = '需人工复核' if need_review else pred_class
    data['raw_class'] = pred_class
    data['confidence'] = round(conf, 4)
    data['need_review'] = need_review
    data['probs'] = [round(p, 4) for p in probs]
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