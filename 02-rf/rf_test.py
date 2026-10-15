import pandas as pd
import pickle
from config import Config

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

config =Config()

#加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


#加载验证集数据
dev_data = pd.read_csv(config.process_dev_path)
print(dev_data.shape)

#提取数据列标签列
words = dev_data['words']
y_test = dev_data['label']

#样本送入tfidf向量化器
features = tfidf_vectorizer.transform(words)
print(features.shape)

y_pred = rf_model.predict(features)


print(f'测试结果:{y_pred[:20]}')
print(f'准确率:{accuracy_score(y_pred,y_test)}')

print('混淆矩阵:',confusion_matrix(y_pred,y_test))

print('精确率macro:',precision_score(y_pred,y_test,average='macro'))
print('精确率micro:',precision_score(y_pred,y_test,average='micro'))
print('精确率weighted:',precision_score(y_pred,y_test,average='weighted'))

print('召回率macro:',recall_score(y_pred,y_test,average='macro'))
print('召回率micro:',recall_score(y_pred,y_test,average='micro'))
print('召回率weighted:',recall_score(y_pred,y_test,average='weighted'))

print('f1-score_macro:',f1_score(y_pred,y_test,average='macro'))
print('f1-score_micro:',f1_score(y_pred,y_test,average='micro'))
print('f1-score_weighted:',f1_score(y_pred,y_test,average='weighted'))

print('评估报告:',classification_report(y_test,y_pred))