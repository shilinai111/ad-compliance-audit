"""
演示
    基于预训练BERT模型进行下游文本分类微调的流程
NLP中神经网络训练（微调）的工作流：

0. 全局配置

1.加载分词器/词表

 1.1  加载bert_tokenizer 并加载 自带词表  （与 “预训练模型 ” 捆绑 ）
     我们使用bert自带的分词器， 不能使用jieba分词器，因为我们不是对bert模型进行 预训练
 1.2  创建BERT配置对象   读取预训练的Bert模型的网络结构参数


2.准备数据集

   2.1 定义函数，加载数据，返回 (文本,标签索引)  元组数据形式。

   2.2 构造样本的数据集对象, MyDataset类  ，  定义

   2.3 定义整理(批处理)函数 collate   （dataloader的核心组件）

3 构建数据加载器

4 搭建神经网络

    预训练BERT，输出last_hidden_state(表示token语义),

    线性层nn.Linear

  5.模型训练  2244    - 训练集 + 验证集

  1.前向传播

  2.计算损失

  3.梯度清零

  4.反向传播

   5.更新参数

   6 记录loss

   7 计算准确率等指标


  5.模型测试
     1.前向传播
    2.计算损失


模型调优思路:
    1.调整超参数：学习率、批次大小、训练轮数、权重衰减等
    2.调整模型结构：增加线性层、增加dropout层、使用不同的预训练模型等


要求：
1  看懂整体逻辑
2  能够读懂每一行代码
"""

# 导包
import os

import json
from pathlib import Path  # 用于把训练结果固定写到脚本所在目录

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关掉 TensorFlow 在cpu上运行时的警告，关闭警告
from tqdm import tqdm  # 可视化训练过程
import torch  # pytorch框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块，提供各种优化器，比如SGD,Adam,AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig  # Bert模型，Bert分词器，基本配置

# huggingface的transformers框架的具体模型方法  pip install transformers -i


'''  演示说明 bert架构，也先分词--> 词库 -->  ID --> 词向量 --  bert模型（语义特征） --> 分类层。 
（目前是预训练）模型训练好之后， 分词器中的词库有多大，每个词的位置索引已确定，
后续BERT看到你的输入的词索引，就知道你输入的是哪个词。

“错配”的例子：
Bert预训练时，"吃苹果" 这个词被拆成 ["吃", "苹", “果"] 并分别赋予了 ID [1111, 2222, 3333]。
但如果你微调时用了一个按词分词的工具，把 "吃苹果" 拆分为 ["吃", "苹果"]，给它 ID [7777, 8888]。

换一个分词器，新词库大小，顺序完全不一样，bert拿到词向量完全不认识， 除非重新训练。
类比：  充电器要和  手机匹配。
'''

from config import Config  # 自定义的全局配置类

# 计算 accuracy准确率 和 F1(精确率和召回率的调和平均)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt  # 绘图


config = Config()

#加载词表
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)

# print('词表',BERT_TOKENIZER.vocab)

#读取配置 获取模型超参
BERT_CONFIG = BertConfig.from_pretrained(config.bert_path)

#准备数据集
#定义函数读取数据
def load_raw_data(file_path):
    '''
        输入：待处理的数据路径
        输出：数据和标签[(文本，标签索引),(文本，标签索引)]
    '''
    #初始化空列表
    results=[]

    #加载数据
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            #文本分割 text和label
            text, label = line.split('\t')
            results.append((text, int(label)))
        return results



#数据集对象生成

class MyDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text,label = self.data_list[idx]
        return text,label


#依据dataloader抽到的样本序号，根据dataset指定的单样本处理规则
# 进行量产（批量生成一个批次样本的数据，将数据打包成一个大张量，送入模型，并行训练）
def collate_fn(batch):
     #分割文本和标签
    texts,labels = zip(*batch)

    #送入分词器
    output = BERT_TOKENIZER(
        list(texts), #一个批次的切词后的文本
        padding = "max_length",
        truncation = True,
        max_length=config.max_len,
        return_tensors='pt',
    )



     #输出张狂
    input_ids = output['input_ids']
    attention_mask = output['attention_mask']
    # print(config.max_len)
    # print(input_ids.shape)
    # print(attention_mask.shape)

    #获取标签labels（兜底转换，防止个别标签仍是字符串/浮点）
    labels = torch.tensor([int(label) for label in labels], dtype=torch.int64)

    return input_ids, attention_mask, labels


def build_dataloaer(train_dataset,dev_dataset,test_dataset):

    #训练集数据加载其
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn, #批处理，将一个批次整理为相同形状的张量
        drop_last = True
    )
    # 验证集数据加载其
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,  # 批处理，将一个批次整理为相同形状的张量
        drop_last=True
    )

    # 测试集数据加载器
    # 【已修复】原代码这里也设了 drop_last=True，会把测试集尾部不满一批的样本丢掉，
    # 导致测试指标统计不全。评估阶段必须看完每一条样本，所以设为 False。
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,  # 批处理，将一个批次整理为相同形状的张量
        drop_last=False
    )
    return train_dataloader,dev_dataloader,test_dataloader

#搭建神经网络
class MyBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        #定义Bert
        self.bert = BertModel.from_pretrained(config.bert_path)

        #输出层
        self.fc = nn.Linear(BERT_CONFIG.hidden_size,len(config.id2class))
    #正向传播
    def forward(self, input_ids,attention_mask):
        #bert输出
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #token_type_ids   单句子情况下自动生成全0向量
            #position_ids  无需 自动生成递增序列
        )
        # output 输出为2个张量
        # 1. last_hidden_state(batch, seq_len, dim)
        #表示编码器最后一层输出的每个token的特征向量用于逐字分类任务
        # 2 .pooler output 池化输出，表示的是整句文本的综合语义(batch, dim)
        #用于整句文本的分类任务
        # 输出层将pooler output转化为 10个类别上的Logits

        # # 方式2 取原生CLS向量（更推荐）
        # cls_emb = output.last_hidden_state[:, 0, :]
        # out = self.dropout(cls_emb)

        logits = self.fc(output.pooler_output)
        return logits

def train_one_epoch(model, train_dataloader, loss_fn, optimizer):

    model.train()

    #初始化每轮损失值
    total_loss = 0

    total_preds = []
    total_label = []

    for batch in tqdm(train_dataloader, desc='Training'):

    # print(batch)  # 输出的是一个元组  input_ids :  索引序列  batch, seq_len
    #                               attention_mask : 注意力遮罩  batch, seq_len
    #                               labels ： 一个批次样本的标签    batch

    #将三个张量送入指定设备
        input_ids ,attention_mask,labels = [item.to(config.device) for item in batch]

        #前向传播
        logits = model(input_ids,attention_mask)

        #计算损失
        loss = loss_fn(logits,labels)

        #梯度清零 反向传播 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #为了模型的准确率和f1值
        #需要获取所有样本的预测类别和标签

        #`logits` shape: `[batch_size, num_classes]`[16,10] -->[16, ]
        y_pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
        # torch.argmax(logits, dim=-1)
        # y_pred 是一个列表，包含了 一个批次16个样本的 预测类别  ，  len=batch size

        total_preds.extend(y_pred)

        #标签的list
        total_label.extend(labels.cpu().numpy().tolist())

    #计算平均损失
    avg_loss = total_loss / len(train_dataloader)

    #计算f1得分
    acc = accuracy_score(total_label,total_preds)
    f1s = f1_score(total_label,total_preds, average='macro')

    return avg_loss, acc, f1s





# 验证函数 + 仅正向传播
def evaluate(model, dev_dataloader, loss_fn):

    model.eval()  # 模型切换到训练状态

    # 初始化每轮的累积损失值
    total_loss = 0.0

    # 定义空的列表
    total_preds=[]   # 每个批次的预测类别
    total_label=[]   # 每个批次的类别标签

    # 训练一个轮次，对其中各个批次进行循环
    for batch in tqdm(dev_dataloader, desc='Evaluating'):

        #print(batch)  # 输出的是一个元组  input_ids :  索引序列  batch, seq_len
        #                               attention_mask : 注意力遮罩  batch, seq_len
        #                               labels ： 一个批次样本的标签    batch

        # 将3个张量送入指定设备
        input_ids, attention_mask, labels = [ item.to(config.device) for item in batch ]


        # 正向传播
        logits = model(input_ids, attention_mask)
        # shape:  batch, num_classes

        # 计算损失
        loss = loss_fn(logits, labels)


        total_loss += loss.item()

        ''' 为了模型的 准确率和 f1值 
        需要获得所有样本上的预测类别和标签'''
        #  shape :  logits （16， 10）  -->   y_pred ( 16, )
        y_pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
                # torch.argmax(logits, dim=-1)
        # y_pred 是一个列表，包含了 一个批次16个样本的 预测类别  ，  len=batch size

        total_preds.extend(y_pred)

        # 标签的list
        total_label.extend(labels.cpu().numpy().tolist())


    #计算平均损失
    avg_loss =total_loss/len(dev_dataloader)

    #在整个训练集上计算准确率 f1得分
    acc = accuracy_score(total_label,total_preds)
    f1s = f1_score(total_label,total_preds, average='macro')

    return avg_loss, acc, f1s












def predict_all(model, dataloader):
    """
    跑完整个数据集，返回 (真实标签列表, 预测标签列表)
    注意这里用 torch.no_grad()，评估阶段不需要构建计算图，能省显存也更快。
    """
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(dataloader, desc='Predicting'):
        input_ids, attention_mask, labels = [item.to(config.device) for item in batch]
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        y_pred.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
        y_true.extend(labels.cpu().numpy().tolist())
    return y_true, y_pred


def report(y_true, y_pred, title='测试集'):
    """
    打印准确率、macro-F1、分类报告和混淆矩阵。
    广告审核场景里混淆矩阵尤其重要：
    把「黑产广告」误判成「不违规」属于严重漏放，必须能从矩阵里一眼看出来。
    """
    names = [config.id2class[i] for i in sorted(config.id2class)]

    print('\n' + '=' * 60)
    print(f'{title}评估结果')
    print('=' * 60)
    print('accuracy :', round(accuracy_score(y_true, y_pred), 4))
    print('macro-F1 :', round(f1_score(y_true, y_pred, average='macro'), 4))
    print()
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print('混淆矩阵（行 = 真实类别，列 = 预测类别）')
    print('真实\\预测'.ljust(12) + ''.join(n.ljust(10) for n in names))
    for i, name in enumerate(names):
        print(name.ljust(12) + ''.join(str(v).ljust(10) for v in cm[i]))


def train(train_dataloader,dev_dataloader,model):
    #损失函数
    # 类别加权（可在 config 里开关）：欺诈/不违规是少数类（测试集各仅 100 条，
    # 而低俗色情 201 条），不加权时模型天然偏向多数类，少数类召回被压低。
    # weight_i = sqrt(N / (K * count_i))，开根号是为了缓和过度补偿。
    if getattr(config, 'use_class_weight', False):
        lb = torch.tensor([int(x[1]) for x in train_dataloader.dataset.data_list])
        counts = torch.bincount(lb, minlength=len(config.id2class)).float()
        weights = (lb.numel() / (len(counts) * counts)).sqrt()
        print('类别权重:', {config.id2class[i]: round(w.item(), 3) for i, w in enumerate(weights)})
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(config.device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    #优化器
    optimizer = optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)

    #初始化训练 指标
    best_f1 = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []

    #对轮数进行循环
    for epoch in range(config.epochs):
        # 对批次进行循环
        train_loss,train_acc,train_f1 = \
            train_one_epoch(model, train_dataloader, loss_fn, optimizer)

        #一轮训练完，评估
        val_loss, val_acc, val_f1 = \
            evaluate(model, dev_dataloader, loss_fn,)

        # 打印一轮训练完的指标
        print('epoch:', epoch,
              'trainloss=', round(train_loss, 4), 'trainAcc=', round(train_acc, 4),
              'trainF1=', round(train_f1, 4),
              'valloss=', round(val_loss, 4), 'valAcc=', round(val_acc, 4),
              'valF1=', round(val_f1, 4))

        # 将每一轮的测评指标存入列表
        # 【已修复】原代码把这几行 append 写在下面的 if 里面，
        # 只有刷新最优的那一轮才记录，画出来的曲线永远只有一个点。
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # 只在验证集 F1 刷新最好成绩时才保存模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            #保存模型
            torch.save(model.state_dict(), config.bert_model_path)
            print('  -> 验证集F1刷新最优，已保存模型')

    results = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "train_f1s": train_f1s,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1s": val_f1s
    }

    # 存到脚本所在目录，避免从不同工作目录运行时文件散落各处
    results_path = os.path.join(Path(__file__).parent, "train_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 【已修复】原函数没有 return，main 里拿到的 results 是 None，
    # 传给 plot_history 会直接报错。
    return results


# 4.4 绘制训练曲线   这个明旋直接从代码中copy过来，不手敲
def plot_history(history):
    # 1.设置x轴刻度，epoch轮次
    epochs = range(1, len(history['train_losses'])+1)  # range包头不包尾
    # 2.设置画布大小  15x5英寸
    plt.figure(figsize=(15, 5))
    # 3.绘制loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], label='train_loss')   #  没有横坐标 也没事
    plt.plot(epochs, history['val_losses'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # 4.绘制acc曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_accs'], label='train_acc')
    plt.plot(epochs, history['val_accs'], label='val_acc')
    plt.title('Acc Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend()
    # 5.绘制f1曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1s'], label='train_f1')
    plt.plot(epochs, history['val_f1s'], label='val_f1')
    plt.title('F1 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.grid(True)
    plt.legend()

    # 优化布局   自动调整子图间距
    plt.tight_layout()

    # 存成图片：无显示器的环境（服务器 / 后台运行）也能留下训练曲线
    save_path = os.path.join(Path(__file__).parent, "train_curves.png")
    plt.savefig(save_path, dpi=120)
    print(f'\n训练曲线已保存：{save_path}')

    plt.show()




if __name__ == '__main__':
    train_data = load_raw_data(config.train_path)
    dev_data = load_raw_data(config.dev_path)
    test_data = load_raw_data(config.test_path)

    print('3份样本数:',len(train_data),len(dev_data),len(test_data))

    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)

    # 数据加载器
    train_dataloader, dev_dataloader, test_dataloader = \
        build_dataloaer(train_dataset, dev_dataset, test_dataset)

    #实例化模型
    model = MyBertClassifier().to(config.device)

    # 模型训练代码
    results = train(train_dataloader, dev_dataloader, model)

    # 加载验证集上表现最好的那一份权重，在测试集上做最终评估
    # （原代码建了 test_dataloader 却从没用过，测试集白准备了）
    model.load_state_dict(torch.load(config.bert_model_path, map_location=config.device))
    y_true, y_pred = predict_all(model, test_dataloader)
    report(y_true, y_pred, title='测试集')

    # 绘制图像
    plot_history(results)
