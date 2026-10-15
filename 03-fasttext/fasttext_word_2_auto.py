"""
演示
    fasttext文本分类训练  (只能在CPU上训练),   采用手动调参，进行词级分词
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
dev_file = ft_ascii(config.word_dev_path)
test_file = ft_ascii(config.word_test_path)

# todo 模型训练
#model = fasttext.train_unsupervised() 用于训练词向量  自监督
model = fasttext.train_supervised(
                input = train_file,   # 训练集路径 按词分词
                autotuneValidationFile=dev_file, # 提供独立的验证集，用于对每一次调参训练后的模型进行性能评估
                verbose = 3, # 输出训练过程中的详细信息 ，当前搜索到的最佳参数组合。
                seed = 10,   # 随机数种子
                autotuneDuration =30,
                # 【已修复】不加约束时自动调参会挑到很大的 dim，
                # 输入矩阵能膨胀到 1000万×dim（十几 GB），直接撑爆内存把进程杀掉。
                # 限定模型体积上限，牺牲一点精度换"能跑完"。
                autotuneModelSize = '50M',
                )
'''哪些参数 ？     
自动调参，不用设置，不用管参数 
参数名称	作用	搜索范围
epoch     1~100
lr      0.01~1
dim    5~500
minCount    默认5  最低词频， 如果某词的出现频率太小，则不进入词库
wordNgrams   1~5
minn         字符 n-gram的最小长度
maxn         字符 n-gram的最大长度
'''


# todo 模型保存（先存临时 ASCII 目录再拷回，fastText 写不了中文路径）
model.save_model(ft_model_tmp('ft_model_word_2.bin'))
ft_save_back(ft_model_tmp('ft_model_word_2.bin'), config.ft_model_path.replace('.bin','_word_2.bin'))

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
# 【已修复】自动调参可能把 dim 调到几百，此时输入矩阵会达到 1000万×dim，
# 光这一个矩阵就要十几 GB，直接撑爆内存把进程杀掉（退出码 1，很难定位）。
# 这行只是教学演示打印，失败不该拖垮整个训练流程，故做保护。
try:
    print(model.get_input_matrix().shape)  #(2004760, 32)
except Exception as e:
    print('get_input_matrix 不可用（不影响训练结果）:', type(e).__name__, e)



