"""
数据的预处理， 进行逐行的jieba分词， 并将结果保存为csv文件
步骤
1.读取原始数据
2逐行进行jieba分词
3获取每行的序列长度
4可视化呈现 序列长度的分布
5保存处理完的结果，另存为CSV
"""
import pandas as pd
import os
import jieba
from config import *
import matplotlib.pyplot as plt # 可视化工具
import seaborn as sns # 可视化工具

config= Config()

def process_data(data_path,save_path):
    #1.读取原始数据
    data=pd.read_csv(data_path,sep='\t',names=['text','label'])
    print(data.head())



#2.逐行进行jieba分词   获取每行序列长度
    def cut_words(sentence):
        # 对文本切词 限制数据规模 超过20截断
        split_word = jieba.lcut(sentence)[:20]
        return ' '.join(split_word),len(split_word)
    data['words'],data['seq_len']=zip(*data['text'].apply(cut_words))

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None):

        print(data)

    #4.可视化呈现 序列长度分部
    data['seq_len'].hist()
    plt.show()
    print(data['seq_len'].describe())


    #5.保存处理完的结果 另存csv
    data.to_csv(save_path,index=False)




if __name__ == '__main__':
    process_data(data_path = config.train_path,save_path = config.process_train_path)
    # process_data(data_path=config.dev_path, save_path=config.process_dev_path)
    # process_data(data_path=config.test_path, save_path=config.process_test_path)