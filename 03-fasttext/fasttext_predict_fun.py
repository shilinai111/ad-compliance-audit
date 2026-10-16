"""
演示
    模型预测,这里采用ft_model_char_1.bin(按字分词+手动调参)
涉及到的API:
    fasttext.load_model
    model.predict

步骤
创建配置对象
加载模型
定义函数，实现模型预测

"""
from unittest import result

# 导包
import fasttext
import time
# 【已修复】load_model 同样受中文路径限制，需要 ft_ascii 把模型搬到临时目录再加载
from config import *


'''  需要打临时补丁修复  否则因为 numpy版本与fasttext版本冲突问题会运行出错  '''
#===================================================
import numpy as np
#需要重新定义 np.array
# 逻辑： 对 np.array() 的调用时，强制将 copy=False 改为 copy=True，
# 然后交给原始的 NumPy 函数去执行。
old_array = np.array
def patched_array(obj, copy=True, *args, **kwargs):
    if copy is False:
        copy = True
    return old_array(obj, copy=copy, *args, **kwargs)
np.array = patched_array
#===================================================



# 创建配置对象
config=Config()

# 模型加载（先拷到纯 ASCII 临时目录，fastText 打不开中文路径）
model= fasttext.load_model(ft_ascii(config.ft_model_path.replace('.bin','_char_1.bin')))

# 推理函数
def predict_fun(data):
     """
     预测函数：返回类别 + 六类概率 + 置信度拒判结果（与 BERT 路线的返回契约保持一致）
     :param data: 传入的字典，格式{text: 文本内容}
     :return: {text, pred_class, raw_class, confidence, need_review, probs}
     """

     # 根据训练时的数据预处理，对客户端送入的数据文本进行按字分词
     text = data['text']

     split_text = ' '.join(list(text))

     # k=类别数：要回全部类别的概率（用来画概率条 + 做拒判），不再只要 Top-1。
     # 之前用的是默认 k=1，概率信息就此丢掉。
     labels, scores = model.predict(split_text, k=len(model.get_labels()))

     with open(config.class_path, 'r', encoding='utf-8') as f:
         class_list = [line.strip() for line in f if line.strip()]

     # fastText 的标签是中文类别名（__label__不违规），且只返回概率非零的类别，
     # 所以要按名字映射回固定顺序的六槽位
     probs = [0.0] * len(class_list)
     for lab, sc in zip(labels, scores):
         name = lab.replace('__label__', '')
         if name in class_list:
             probs[class_list.index(name)] = float(sc)

     top = max(range(len(probs)), key=lambda i: probs[i])
     y_pred = class_list[top]
     conf = probs[top]

     # 与 BERT 路线一致的拒判语义：低于阈值时 pred_class 换成「需人工复核」，
     # 原判保留在 raw_class 供人工参考
     threshold = getattr(config, 'conf_threshold', 0.0)
     need_review = threshold > 0 and conf < threshold

     data['pred_class'] = '需人工复核' if need_review else y_pred
     data['raw_class'] = y_pred
     data['confidence'] = round(conf, 4)
     data['need_review'] = need_review
     data['probs'] = [round(p, 4) for p in probs]

     return data

     # print(y_pred)





if __name__ == '__main__':
     # 换成广告审核场景的样本：覆盖正常文案 / 黑产引流 / 诈骗话术 / 博彩域名
     for t in ['本款洗面奶温和清洁，敏感肌可用',
               '店铺好评，销量提升联系我/我最美',
               '不用出门一天轻轻松松二三百，不收费需要滴滴',
               '浏览器搜索yhh09.top']:
          print(predict_fun({'text': t}))












