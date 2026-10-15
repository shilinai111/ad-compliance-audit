"""
演示
    fasttext文本分类训练  (只能在CPU上训练),   采用手动调参，进行jieba分词
涉及到的API:
    fasttext.train_supervised()

    注意： 路径中不能有中文字符

步骤
创建配置对象

模型训练

模型保存

模型评估

打印信息
"""

import fasttext
# 【已修复】原来写的是 from config import Config，拿不到中文路径兼容工具 ft_ascii
from config import *


# todo 创建配置对象
config = Config()

# 中文路径兼容：把数据搬到纯 ASCII 临时目录（详见 config.py 的说明）
train_file = ft_ascii(config.word_train_path)
test_file = ft_ascii(config.word_test_path)

# todo 模型训练
#model = fasttext.train_unsupervised() 用于训练词向量  自监督
model = fasttext.train_supervised(
                input = train_file,   # 训练集路径 按词分词
                dim=32,      # 稠密词向量的维度
                wordNgrams=2, # n-gram特征，默认为1。设置=2：使用二元组词语+一元组
                epoch=100,     # 训练轮数
                lr = 0.01      # 学习率
                )

# todo 模型保存（先存临时 ASCII 目录再拷回，fastText 写不了中文路径）
model.save_model(ft_model_tmp('ft_model_word_1.bin'))
ft_save_back(ft_model_tmp('ft_model_word_1.bin'), config.ft_model_path.replace('.bin','_word_1.bin'))

# todo 模型评估  用测试集进行评估    默认使用micro 平均
result = model.test(test_file)
# 返回3个数据
print('测试集 -> 样本数， 精准率P，  召回率R',  result)

# 计算f1 得分
f1_score = 2*result[1]*result[2]/(result[1]+result[2]) if (result[1]+result[2])>0 else 0
print('测试集 F1:', round(f1_score, 4))

# todo 打印信息
# 获取指定词的词向量
print('词向量', model.get_word_vector('中'))
print('词向量', model.get_word_vector('中 华'))
print('词向量', model.get_word_vector('中华人民共和国'))

#从词库中提取词表
print('词库', model.get_words(include_freq=True))   # 4760

# 词库大小
print(len(model.get_words()))

# 展示第一个线性层的权重矩阵
# 【已修复】fastText 从 C++ 侧取回大矩阵时偶发崩溃，且不抛 Python 异常
# （进程直接以退出码 1 结束）。这行只是教学演示打印，失败不该拖垮训练流程。
try:
    print(model.get_input_matrix().shape)  #(2004760, 32)
except Exception as e:
    print('get_input_matrix 不可用（不影响训练结果）:', e)



