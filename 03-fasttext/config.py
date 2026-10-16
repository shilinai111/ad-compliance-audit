"""
    配置文件，统一管理全局配置信息，包括数据路径、类别信息等。方便项目的管理和后期维护。
    实现功能： 对数据集 xx_raw.txt 进行去重，保存为新路径
"""
import os   # 文件操作模块
from pathlib import Path  # 路径操作模块
import shutil
import tempfile
# ==========================================================================
# fastText 中文路径兼容层
# --------------------------------------------------------------------------
# fastText 是 C++ 实现，Windows 下用**窄字符（ANSI）**路径打开文件。
# 一旦项目路径含中文（本项目是 D:\pythondm\深度学习\code\...），
# train_supervised / load_model 都会报 "cannot be opened for training"。
# 原项目能跑是因为它当时的路径是纯 ASCII 的。
# 解决办法：读写前把文件搬到纯 ASCII 的临时目录，用完再拷回项目目录。
# （Python 自己的 open() 不受影响，只有 fastText 的 C++ 内核受影响。）
# ==========================================================================
FT_ASCII_DIR = os.path.join(tempfile.gettempdir(), "ft_ascii_workspace")
os.makedirs(FT_ASCII_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# fastText 与 numpy 2.x 的兼容补丁
# --------------------------------------------------------------------------
# fastText 内部用 np.array(..., copy=False) 取回词向量/权重矩阵，
# numpy 2.x 已经禁止 copy=False 语义，调用 get_input_matrix() 会直接崩，
# 而且不抛 Python 异常（进程以退出码 1 结束），排查起来非常费劲。
# 原项目只在 fasttext_predict_fun.py 里打过这个补丁，训练脚本没打，
# 于是 4 个训练脚本里有 1 个会静默崩溃。这里统一处理：只要 import config 就生效。
import numpy as np

_np_array_original = np.array


def _np_array_copy_true(obj, copy=True, *args, **kwargs):
    """强制 copy=True，绕开 numpy 2.x 对 copy=False 的限制。"""
    return _np_array_original(obj, copy=True, *args, **kwargs)


np.array = _np_array_copy_true


def ft_ascii(src_path):
    """把数据文件复制到纯 ASCII 临时目录，返回可直接传给 fastText 的路径。"""
    dst = os.path.join(FT_ASCII_DIR, os.path.basename(src_path))
    if os.path.abspath(src_path) != os.path.abspath(dst):
        shutil.copyfile(src_path, dst)
    return dst


def ft_model_tmp(name):
    """返回模型在临时目录中的保存路径（fastText 无法直接写中文路径）。"""
    return os.path.join(FT_ASCII_DIR, name)


def ft_save_back(tmp_model_path, dst_path):
    """把临时目录里训好的模型拷回项目目录。"""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(tmp_model_path, dst_path)
    print('模型已保存：', dst_path)
    return dst_path


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
        self.dev_path= os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 停用词文件路径
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 3.初始化类别文件路径
        self.class_path = os.path.join(self.root_path, "class.txt")

        self.process_train_path = os.path.join(self.root_path, "process_train.csv")
        self.process_dev_path = os.path.join(self.root_path, "process_dev.csv")
        self.process_test_path = os.path.join(self.root_path, "process_test.csv")

        #创建新文件夹，保存每个模型训练后的参数文件
        os.makedirs(os.path.join(self.root_path, "save_model"), exist_ok=True)

        #训练时的路径
        self.rf_model_path = os.path.join(self.root_path, "save_model",'rf_model.pkl')

        #保存tf-idf算法的结果’
        self.tfidf_model_path = os.path.join(self.root_path, "save_model",'tfidf_model.pkl')

        #API服务搭建参数配置
        self.api_host = '127.0.0.1'
        # 8002：避开 Chromium 受限端口 6000（浏览器会报 ERR_UNSAFE_PORT），
        # 并与 BERT(8000) / RF(8001) 区分，三个服务可同时启动。
        self.api_port = '8002'  #端口号

        # 置信度阈值：与 BERT 路线同名同义，低于该值转人工复核（设为 0 即关闭该功能）。
        # fastText 的概率是 6 类上的归一化得分，尺度比随机森林的投票比例更接近 softmax。
        self.conf_threshold = 0.60

        '''Fasttext的路径设置'''
        #按字切词
        self.char_train_path = os.path.join(self.root_path, "char_train.txt")
        self.char_dev_path = os.path.join(self.root_path, "char_dev.txt")
        self.char_test_path = os.path.join(self.root_path, "char_test.txt")

        #按词切词
        self.word_train_path = os.path.join(self.root_path, "word_train.txt")
        self.word_dev_path = os.path.join(self.root_path, "word_dev.txt")
        self.word_test_path = os.path.join(self.root_path, "word_test.txt")

        '''fasttext模型保存的路径'''
        self.ft_model_path = os.path.join(self.root_path, "save_model",'ft_model.bin')

        #创建一个索引 和分类名称之间的映射表
        #读取类别数据讲预测的类别索引 转回文字
        with open(self.class_path,'r',encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]

        #生成映射表
        self.id2class = {i:name for i ,name in enumerate(class_list)}

# 2. 对数据文件进行去重并保存到新的文件中
def remove_duplicates(input_file, output_file):
    """
    读取文件去重并保存。
    """
    seen = set()    # 集合    无序、自带去重、查询快
    total_lines = 0  #  原样本总数
    unique_lines = 0 #  去重后样本数

    # 确保输出目录存在，避免保存报错
    # os.path.dirname 父路径
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 读取文件，去重并写入新文件
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            total_lines += 1   # 总行数+1
            if line not in seen:       #判断是否存在
                outfile.write(line)    #写入文件
                seen.add(line)         #追加
                unique_lines += 1

    print(f"去重前总行数: {total_lines}")
    print(f"去重后保留行数: {unique_lines}")
    print(f"重复行数: {total_lines - unique_lines}")
    print("-" * 30)

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
    remove_duplicates(config.train_raw_path, config.train_path)
    remove_duplicates(config.dev_raw_path, config.dev_path)
    remove_duplicates(config.test_raw_path, config.test_path)
