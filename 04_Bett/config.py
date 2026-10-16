"""
    配置文件，统一管理全局配置信息，包括数据路径、类别信息等。方便项目的管理和后期维护。

    业务场景：广告素材投放前的合规审核（违规内容识别 + 拦截）
    数据来源：HuggingFace  zjunlp/ChineseHarm-bench
    类别：不违规 / 黑产广告 / 欺诈 / 博彩 / 谩骂引战 / 低俗色情  共 6 类

    数据集由 01-data/prepare_adaudit_data.py 生成，格式为  文本\t标签索引
    原新闻分类数据（THUCNews）已备份到 01-data/_backup_news/
"""
import os   # 文件操作模块
from pathlib import Path  # 路径操作模块

import torch


# 1.定义配置类，集中管理全局配置信息，比如数据路径、类别信息
class Config():

    # 1.定义__init__方法，初始化配置项
    def __init__(self):
        # 1.初始化根路径
        '''
        __file__：当前文件的完整路径   特殊变量命名
        Path(__file__)：将路径转换为Path对象
        .parent：获取当前文件所在的目录'''
        self.root_path = os.path.join(Path(__file__).parent.parent, "01-data")
        # self.root_path = r"D:\tmf_project_shenzhen_AI_4\07_live_code\01-data"
        # 2.初始化数据文件路径 （raw  有些样本完全一致）
        self.train_raw_path = os.path.join(self.root_path, "train_raw.txt")
        self.dev_raw_path = os.path.join(self.root_path, "dev_raw.txt")
        self.test_raw_path = os.path.join(self.root_path, "test_raw.txt")
        # 去重后数据集路径
        self.train_path = os.path.join(self.root_path, "train.txt")
        self.dev_path = os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 停用词文件路径
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 3.初始化类别文件路径
        self.class_path = os.path.join(self.root_path, "class.txt")

        # 新增一些属性 ，包含（经过jieba分词后需要保存的额数据）路径
        self.process_train_path = os.path.join(self.root_path, "process_train.csv")
        self.process_dev_path = os.path.join(self.root_path, "process_dev.csv")
        self.process_test_path = os.path.join(self.root_path, "process_test.csv")

        # 创建新文件夹，用于保存每个模型训练后的参数文件
        os.makedirs( os.path.join(self.root_path, "save_model"), exist_ok=True  )

        # 训练时的路径
        self.rf_model_path = os.path.join(self.root_path, "save_model", "rf_model.pkl")

        # 保存 tf-idf算法的结果
        self.tfidf_model_path = os.path.join(self.root_path, "save_model", "tfidf_model.pkl")



        '''  Fasttext 的路径设置 '''
        # 按字切词
        self.char_train_path = os.path.join(self.root_path, "char_train.txt")
        self.char_dev_path = os.path.join(self.root_path, "char_dev.txt")
        self.char_test_path = os.path.join(self.root_path, "char_test.txt")

        # 按词切词 （jieba）
        self.word_train_path = os.path.join(self.root_path, "word_train.txt")
        self.word_dev_path = os.path.join(self.root_path, "word_dev.txt")
        self.word_test_path = os.path.join(self.root_path, "word_test.txt")

        '''  fasttext模型保存的路径 '''
        self.ft_model_path = os.path.join(self.root_path, "save_model", "ft_model.bin")

        # 创建一个 索引 和 分类名称之间的映射表
        # 读取 类别数据，将预测的类别索引，转回文字
        with open(self.class_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]

        # 生成映射表
        self.id2class = { i: name  for i , name in  enumerate(class_list)  }





        #  API服务搭建的参数配置
        self.api_host = '127.0.0.1'  # 代表 将服务部署在本机 并只能由本机访问
        # 【重要】端口不能用 6000：6000 在 Chromium（Chrome / Edge）的受限端口名单里，
        # 浏览器会直接拒绝连接并报 ERR_UNSAFE_PORT —— 前端页面将永远连不上模型。
        # 注意 curl / requests 不检查这份名单，所以命令行测试会"看起来一切正常"。
        # 这里改用 8000（浏览器安全端口），并与 RF(8001) / fastText(8002) 区分开，避免互相抢端口。
        self.api_port = '8000'


        #  bert 新增
        # Bert模型的存放路径
        self.bert_path = os.path.join(self.root_path, "bert-base-chinese")

        #Bert微调后存储位置
        self.bert_model_path = os.path.join(self.root_path, "save_model", "bert_model.pth")

        # 设定硬件：有显卡用显卡，没有就退回 CPU
        # 【已修复】原代码在下面又写了一次 self.device = torch.device('cpu')，
        # 把这里的判断覆盖掉了，导致训练永远跑在 CPU 上。现在拆成两个独立变量。
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 超参数：训练前设置的参数，训练过程中不能学习的参数
        self.epochs = 3       # 本数据集约 9.5k 条，远小于原新闻集，需要多训几轮才收敛
        # 本机是 RTX 3050 Ti Laptop（仅 4GB 显存），且浏览器/微信等常驻程序会常驻占用显存。
        # 实测 batch_size=16 时显存吃满触发 WDDM 换页，速度反而暴跌到 1 it/s 以下，故降到 8。
        self.batch_size = 8
        self.max_len = 64     # 文本中位数 19 字、平均 42 字，64 可覆盖绝大多数样本
        self.lr = 2e-5        # BERT 微调的常用学习率（原项目写的 2e-4 偏高，容易训飞）
        # 神经网络训练中常用的L2正则化方法是如何实现的
        self.weight_decay = 1e-2

        # 置信度阈值：广告审核场景下，模型没把握时不应该硬判，
        # 低于该阈值就转人工复核。设为 0 可关闭该功能（退化为纯 argmax）。
        self.conf_threshold = 0.60

        # 类别加权损失：针对「欺诈」类样本少、召回偏低的问题（加权实验后按结果取舍）
        self.use_class_weight = True

        # 动态量化只支持 CPU，单独指定设备，不再覆盖 self.device
        self.quant_device = torch.device('cpu')

        # 保存量化后的模型
        # 【已修复】原文件名拼写错误 quuantizaiton -> quantization
        self.bert_quantization_model_path = os.path.join(self.root_path, "save_model", "bert_quantization_model.bin")

        # ============ 知识蒸馏（Knowledge Distillation）============
        # Teacher = 上面的 12 层 BERT；Student = 层数更少的同构 BERT。
        # Student 不随机初始化，而是从 Teacher 等间隔抽层（见 bert_distillation.py 注释），
        # 这样离线没有小模型也能训，且收敛远快于从头学。
        self.student_layers = 4              # Student 的 Transformer 层数
        self.student_layer_strategy = 'uniform'  #uniform=等间隔抽层 / first=取前N层 / last=取后N层
        self.distill_temperature = 4.0       # 温度 T：越大 soft label 越平滑，暗知识暴露越多
        # alpha 是硬标签 CE 的权重，soft KL 为 (1-alpha)。
        # 【实测调过】alpha=0.5 时 Student 的欺诈召回只有 0.77，
        # 而 Teacher 加权后是 0.94 —— 因为 KL 项不带类别权重，
        # 占比一半就把少数类的判别拉平了，把上一步加权换来的收益弄丢。
        # 提到 0.7 后硬损失（带类别权重）重新占主导，少数类召回跟着回来。
        self.distill_alpha = 0.7             # 硬标签 CE 的权重，soft KL 权重为 (1-alpha)
        self.distill_epochs = 5
        self.bert_student_model_path = os.path.join(self.root_path, "save_model", "bert_student_model.pth")

        # ============ 非结构化剪枝（Unstructured Pruning）============
        # 只剪 nn.Linear 的 weight（BERT 里占参数量 99% 以上），LayerNorm/Embedding 不动。
        # 采用渐进式（gradual pruning）：分多步剪到目标稀疏度，每步剪完微调一轮，
        # 比"一步剪到 50%"的精度损失小得多。
        self.prune_sparsity = 0.5            # 目标全局稀疏度（被剪掉权重的比例）
        self.prune_steps = 3                 # 分几步剪到目标
        self.prune_finetune_epochs = 1       # 每一步剪完后微调几轮
        self.bert_pruned_model_path = os.path.join(self.root_path, "save_model", "bert_pruned_model.pth")






# 2.测试
if __name__ == "__main__":
    # 1.创建配置对象
    config = Config()
    # 2.打印配置信息，验证是否正确加载
    print("根路径:", config.root_path)
    print("训练数据路径:", config.train_path)
    print("验证数据路径:", config.dev_path)
    print("测试数据路径:", config.test_path)
    print("停用词文件路径:", config.stopwords_path)
    print("类别文件路径:", config.class_path)
    # 去重 对 训练、验证、测试  数据集

