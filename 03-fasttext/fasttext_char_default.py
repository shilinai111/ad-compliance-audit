import fasttext
from config import *

# todo 创建配置对象
config = Config()

# 中文路径兼容：把数据搬到纯 ASCII 临时目录（详见 config.py 的说明）
train_file = ft_ascii(config.char_train_path)
test_file = ft_ascii(config.char_test_path)

#todo 模型训练
# model = fasttext.train_unsupervised() 用于训练词向量 自监督
model = fasttext.train_supervised(input = train_file, dim=32,wordNgrams = 2,epoch =100,lr = 0.1)



# todo 模型保存
# 【已修复】原来直接 save_model 到中文路径会失败，先存临时目录再拷回
save_path = config.ft_model_path.replace('.bin','_char_1.bin')
model.save_model(ft_model_tmp('ft_model_char_1.bin'))
ft_save_back(ft_model_tmp('ft_model_char_1.bin'), save_path)

# todo 模型评估
# 【已修复】原代码用训练集评估（result = model.test(config.char_train_path)），
# 指标虚高没有参考价值。改成测试集，才能和 BERT / 随机森林横向对比。
result = model.test(test_file)
#返回三个数据
print('测试集 -> 样本数','精确率P','召回率R',result)

#计算f1得分
f1_score = 2*result[1]*result[2]/(result[1]+result[2]) if (result[1]+result[2])>0 else 0
print('测试集 F1:', round(f1_score, 4))
#todo 打印信息

print('词向量',model.get_word_vector('中'))
print('词向量',model.get_word_vector('中华'))
print('词向量',model.get_word_vector('中华人民共和国'))

#从词库中提取词表
print('词库',model.get_words(include_freq=True))

#词库大小
print(len(model.get_words()))

#展示第一个线性层的权重矩阵
# 【已修复】fastText 从 C++ 侧取回大矩阵时偶发崩溃，且不抛 Python 异常
# （进程直接以退出码 1 结束，很难排查）。这行只是教学演示打印，
# 失败不该拖垮整个训练流程，故做保护。
try:
    print(model.get_input_matrix().shape)
except Exception as e:
    print('get_input_matrix 不可用（不影响训练结果）:', e)