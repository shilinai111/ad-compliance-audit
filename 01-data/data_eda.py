"""
EDA(Exploratory Data Analysis):
    探索性数据分析,帮助发现数据规律或异常，一般集合图标等可视化形式
    就是分析数据，来查看标签数量分布、句子长度分布
"""
# 导包
import pandas as pd # 处理表格数据
from collections import Counter # 统计标签的频次， 是一个字典类
import matplotlib.pyplot as plt # 可视化工具
import seaborn as sns # 可视化工具
from config import Config # 导入配置类

# 1.初始化配置类
config = Config()


# 2.读取 txt 数据  用 tab分割， 存为列表格式，   names为列名：表头
train_data = pd.read_csv(config.train_path, sep="\t", names=["text", "label"])  # dataframe
dev_data = pd.read_csv(config.dev_path, sep="\t", names=["text", "label"])
test_data = pd.read_csv(config.test_path, sep="\t", names=["text", "label"])
print(train_data.shape) # (179k, 2)
print(dev_data.shape)   # (9993, 2)
print(test_data.shape)  # (9990, 2)
print("-"*40)
# 3.统计标签分布
lable_count = Counter(train_data['label'])
print('训练集标签分布', lable_count)

lable_count2 = Counter(dev_data['label'])
print('验证机标签分布', lable_count2)

lable_count3 = Counter(test_data['label'])
print('测试集标签分布', lable_count3)


# 将每一项打印出来



# 4.统计句子长度，默认一个字符是一个token,没有进行分词
# 计算每一行的样本中的汉字数目 并存为新的一列特征

#str字符串访问器
#.replace方法 将字符串中的空格转为空字符串  ’你好    吗‘->’你好吗‘
#.len()获取文本长度
train_data['text_length']=train_data['text'].str.replace(' ','').str.len()
print(train_data.head())

dev_data['text_length']=dev_data['text'].str.replace(' ','').str.len()
print(dev_data.head())

test_data['text_length']=test_data['text'].str.replace(' ','').str.len()
print(test_data.head())


# 5.绘制标签计数图  看每个类别（离散类别）的样本计数
# sns.countplot(x = 'label', data = train_data)
# plt.show()

# sns.countplot(x = 'label', data = dev_data)
# plt.show()
#
# sns.countplot(x = 'label', data = test_data)
# plt.show()

# 6.绘制句子长度直方图， 连连续变量的各个数值区间的数目统计
# 参1 数据集  2 是否要核密度函数拟合分部  3 柱子数
# sns.histplot(train_data['text_length'],kde=False,bins=20)
# plt.show()
#
# sns.histplot(dev_data['text_length'],kde=False,bins=20)
# plt.show()
#
# sns.histplot(test_data['text_length'],kde=False,bins=20)
# plt.show()
# 7. 对测试集进行分析


