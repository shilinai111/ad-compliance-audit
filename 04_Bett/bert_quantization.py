'''
演示
    bert模型的 动态量化DQ
需要掌握的API:
    torch.quantization.quantize_dynamic()
    只能在CPU上使用，不能在GPU上使用  需要提前设定硬件
注意：
    新建一个虚拟环境nlp_cpu, conda create -n nlp_cpu python=3.10
    安装cpu版本的pytorch, pip3 install torch torchvision


  步骤:
# 1 创建配置对象
# 2 创建数据加载器
# 3 加载模型和预训练权重，权重参数为默认的float32
# 4 评估原始模型
# 5 进行动态量化DQ
# 6 评估量化后的模型性能
# 7 保存量化后的模型
'''

import torch
import torch.nn as nn
from config import Config
from bert_train import MyBertClassifier, evaluate, build_dataloaer,load_raw_data, MyDataset


# 1 创建配置对象
config = Config()

# 【必须】动态量化只支持 CPU，这里强制把运行设备切回 CPU。
# config.device 现在会自动识别 CUDA（有显卡就上显卡），如果不强制覆盖，
# evaluate() 会把数据搬到 cuda，而量化后的模型在 cpu，直接报 device 不一致。
#
# 注意：evaluate() / build_dataloaer() 定义在 bert_train.py 里，用的是那个模块级的
# config 实例，不是本文件的这个。所以必须连 bert_train 里的那个一起改，只改本文件没用。
import bert_train as _bert_train
config.device = torch.device('cpu')
_bert_train.config.device = torch.device('cpu')

# 2 创建数据加载器
train_data = load_raw_data(config.train_path)
dev_data = load_raw_data(config.dev_path)
test_data = load_raw_data(config.test_path)

print('3份样本数:', len(train_data), len(dev_data), len(test_data))

# 数据集对象
train_dataset = MyDataset(train_data)
dev_dataset = MyDataset(dev_data)
test_dataset = MyDataset(test_data)
train_dataloader, dev_dataloader, test_dataloader = build_dataloaer(train_dataset, dev_dataset, test_dataset)

# 3 加载模型和预训练权重，权重参数为默认的float32
model = MyBertClassifier()
model.load_state_dict( torch.load(config.bert_model_path, map_location=torch.device('cpu')))

# 4 评估原始模型
model.eval()

test_loss, test_acc,test_f1 = evaluate(model, test_dataloader, loss_fn=nn.CrossEntropyLoss())
print('原始BERT的性能', test_loss, test_acc,test_f1)

# 5 进行动态量化DQ
quantized_model = torch.quantization.quantize_dynamic(
                    model,   # 指明哪个模型需要量化
                    {nn.Linear},  # 指定模型中所有的线性层执行8bit量化 （bert qkv向量的3个线性层 ，前馈神经网络层， 输出层， 多头注意力的上下文融合的线性层）
                    dtype=torch.qint8      )


# 6 评估量化后的模型性能
test_loss2, test_acc2,test_f12 = evaluate(quantized_model, test_dataloader, loss_fn=nn.CrossEntropyLoss())
print('量化BERT的性能', test_loss2, test_acc2,test_f12)


# 7 保存量化后的模型
torch.save(quantized_model.state_dict(), config.bert_quantization_model_path)
print('模型已保存:', config.bert_quantization_model_path)


# 8 量化收益核算：模型体积 + 单条推理耗时
import os, time

def _size_mb(path):
    return os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0.0

def _latency(m, repeat=30):
    """在 CPU 上单条推理，取平均耗时（毫秒）"""
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    sample = test_data[0][0]
    output = tokenizer(
        sample,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = output['input_ids'].to(config.device)
    attention_mask = output['attention_mask'].to(config.device)
    with torch.no_grad():
        for _ in range(3):   # warmup
            m(input_ids, attention_mask)
        t = time.time()
        for _ in range(repeat):
            m(input_ids, attention_mask)
        return (time.time() - t) / repeat * 1000

print()
print('=' * 56)
print('量化收益核算（CPU 环境）')
print('=' * 56)
old_mb, new_mb = _size_mb(config.bert_model_path), _size_mb(config.bert_quantization_model_path)
print(f'原始模型体积 : {old_mb:.1f} MB')
print(f'量化模型体积 : {new_mb:.1f} MB  (压缩到 {new_mb/old_mb*100:.1f}%)' if old_mb else '量化模型体积: -')
t1, t2 = _latency(model), _latency(quantized_model)
print(f'原始单条推理 : {t1:.1f} ms')
print(f'量化单条推理 : {t2:.1f} ms  (加速 {t1/t2:.2f}x)' if t2 else '量化单条推理: -')
print(f'准确率变化   : {test_acc:.4f} -> {test_acc2:.4f}')
print(f'F1 变化      : {test_f1:.4f} -> {test_f12:.4f}')
