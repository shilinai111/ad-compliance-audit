import jieba
from config import *

config = Config()

def process_data(data_path,save_path,is_char=True):
    with open(data_path, 'r', encoding='utf-8') as f:
        #打开输出文件
        with open(save_path, 'w', encoding='utf-8') as fw:
            #读取每一行
            for line in f:
                #去掉收尾空白
                line = line.strip()

                text,label = line.split('\t')
                #标签转为类别名
                label_name = config.id2class[int(label)]

                if is_char:
                    #按字切词
                    split_text = ' '.join(list(text))

                else:
                    #按词切词
                    split_text = ' '.join(jieba.lcut(text))

                #构建fasttext所需要的格式
                '''
                __label__不违规 可 有 偿 助 力 服 务 (char_train.txt 的六分类，后面是按字符分词)
                __label__不违规 有偿 助力 服务 (word_train.txt 的六分类，后面是 jieba 分词)
                '''

                fasttext_line = '__label__'+label_name+' '+split_text+'\n'

                #写入处理后的数据到文件
                fw.write(fasttext_line)

if __name__ == '__main__':
    #训练集
    process_data(config.train_path,config.char_train_path,is_char=True)           #字
    process_data(config.train_path, config.word_train_path, is_char=False)        #词

    #验证集
    process_data(config.dev_path,config.char_dev_path,is_char=True)
    process_data(config.dev_path, config.word_dev_path, is_char=False)

    #测试集
    process_data(config.test_path,config.char_test_path,is_char=True)
    process_data(config.test_path, config.word_test_path, is_char=False)