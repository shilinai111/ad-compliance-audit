import time
import torch
from config import *
from bert_train import MyBertClassifier
from transformers import BertTokenizer

config = Config()

model = MyBertClassifier().to(config.device)

model.load_state_dict(torch.load(config.bert_model_path,map_location=config.device))


BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)

def predict_fun(data):
    model.eval()

    text = data['text']

    output = BERT_TOKENIZER(
        [text],   #输入的必须是可迭代对象
        padding = 'max_length',
        truncation = True,  #截断
        max_length=config.max_len
    )
    #output包含三个序列 input_ids   attention_mask  tokrn_type_ids

    #bert模型预测
    # 推理阶段不需要反向传播，用 no_grad 关掉计算图，更快也更省显存
    with torch.no_grad():
        logits = model(
            input_ids = torch.tensor(output.input_ids).to(config.device),
            attention_mask=torch.tensor(output.attention_mask).to(config.device),
            #token_type_ids 默认生成全0张量
            #position_ids 默认[0,1,2,3]
        )

    # 把 logits 转成概率，用来衡量模型有多"确定"
    probs = torch.softmax(logits, dim=-1)[0]
    conf, y_pred = torch.max(probs, dim=-1)
    conf, y_pred = conf.item(), y_pred.item()

    y_pred_class = config.id2class[y_pred]

    # 广告审核的兜底机制
    # argmax 永远会挑一个类别出来，但真实审核场景里"我没把握"本身就是合法答案。
    # 置信度低于阈值就转人工，避免模型硬判导致误杀正常素材或漏放违规内容。
    threshold = getattr(config, 'conf_threshold', 0.0)
    need_review = threshold > 0 and conf < threshold

    data['pred_class'] = '需人工复核' if need_review else y_pred_class
    data['raw_class'] = y_pred_class      # 保留模型原本的判断，给人工参考
    data['confidence'] = round(conf, 4)
    data['need_review'] = need_review
    return data

if __name__ == '__main__':
    samples = [
        '加我微信，日赚500，操作简单，秒到账，名额有限',
        '本款洗面奶温和清洁，敏感肌可用，已通过国家化妆品备案',
        '今天天气不错，适合出门散步',
    ]
    for s in samples:
        r = predict_fun({'text': s})
        flag = '转人工' if r['need_review'] else '自动判定'
        print(f"[{flag}] {s[:22]:<24} -> {r['pred_class']:<8} 置信度 {r['confidence']}")

    # 测一下单次推理耗时
    data = {'text': samples[0]}
    start = time.time()
    for i in range(100):
        predict_fun(data)
    end = time.time()
    print('单次推理时间:', (end - start) / 100. * 1000, 'ms')
