import pandas as pd
import pickle

# 【已修复】原文件里有一行 `from pygments.lexer import words`，
# 是 IDE 自动补全误插进来的脏导入，完全没用到，却会让脚本一启动就 ModuleNotFoundError。
# 下面第 21 行会重新定义 words 变量，删掉这行没有任何影响。
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from config import Config
from sklearn.ensemble import RandomForestClassifier

config= Config()


#todo # 1.加载数据集
train_dataset = pd.read_csv(config.process_train_path)[:20000]
# print(train_dataset.head())

#todo # 2.文本数值化:使用TF-IDF提取文本特征
#todo #     #2.1提取特征列words(分词结果)
words = train_dataset['words']
#todo #     #2.2 提取标签列
labels = train_dataset['label']
#todo #     #2.3 读取停用词文件，返回停用词列表
with open(config.stopwords_path,'r',encoding='utf-8') as f:
    stopwords = [line.strip() for line in f if line.strip()]
# print(stopwords,len(stopwords))

# todo #     #2.4创建TF-IDF向量化器对象，设置停用词参数
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)

# 如果希望 tfidf向量化器在计算的时候不要忽略单个汉字，在实例化的时候 添加 正则项
# tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")

#todo #     #2.5 使用TF-IDF向量化器对象，对words进行向量化得到向量化矩阵

#fit训练 构建词库 计算TF IDF值
features = tfidf_vectorizer.fit_transform(words)
# print(type(features),features.shape)
# print(features[:5])

#查看tfidf收录的词语

# words_ = tfidf_vectorizer.get_feature_names_out()
# idx = [4045,13322,13805]
#
# for i in idx:
#     print(words_[i])

#todo # 3.训练随机森林模型
#todo #     # 3.1 划分训练集和测试集,使用train_test_split函数，test_size=0.2
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=9,stratify=labels)
print(x_train.shape,x_test.shape)
#todo #     #3.2创建随机森林榄型对象
#参数1随心及森林基学习器的数量 参数2 使用cpu数量
rf_model = RandomForestClassifier(n_estimators=200,n_jobs=-1)
#todo #     #3.3 模型训练
rf_model.fit(x_train,y_train)
#todo # 4.模型评估
y_pred = rf_model.predict(x_test)
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

#todo # 5.保存模型
with open(config.rf_model_path,'wb') as f:
    pickle.dump(rf_model,f)

#保存OF-IDF向量化器
with open(config.tfidf_model_path,'wb') as f:
    pickle.dump(tfidf_vectorizer,f)