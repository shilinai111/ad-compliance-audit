'''
    知识蒸馏（Knowledge Distillation）

    ── 在干什么 ──────────────────────────────────────────────
    让一个「小模型 Student」去模仿「大模型 Teacher」的输出，而不是直接学硬标签。
    Teacher 这里就是 bert_train.py 训好的 12 层 BERT（macro-F1 0.934）。
    Student 是一个只有 4 层的同构 BERT。

    ── 为什么要蒸馏，而不直接训小模型 ──────────────────────────
    硬标签只告诉模型"这条是欺诈"，
    而 Teacher 的 soft label 还告诉它"这条像欺诈 0.7、像博彩 0.25、像不违规 0.05"。
    这个类别间的相似度分布叫「暗知识（dark knowledge）」，
    正是小模型容量不足时最缺的信息。所以同样的小模型，蒸馏往往比直接训高 1~3 个点。

    ── Student 怎么初始化（离线环境的关键）──────────────────────
    本机没有现成的小中文 BERT（只有 bert-base-chinese），也不方便联网下载。
    所以采用「层裁剪初始化」：从 Teacher 的 12 层里挑 4 层，权重原样搬过来。
      - uniform（默认）：等间隔取 [0,3,6,9]，兼顾底层语法特征与高层语义特征
      - first         ：取前 4 层，保留底层特征，丢掉高层语义
      - last          ：取后 4 层，保留高层语义，丢掉底层特征
    比随机初始化收敛快得多，通常也能高 1~2 个点。

    ── 损失函数 ──────────────────────────────────────────────
    L = alpha * CE(student_logits, 真实标签)          # 硬损失：别跑偏
      + (1-alpha) * KL(softmax(z_s/T) || softmax(z_t/T)) * T^2   # 软损失：学暗知识

    T（温度）越大，Teacher 输出的概率分布越平滑，类别间的相对关系越明显。
    乘 T^2 是因为 KL 的梯度会随 1/T^2 缩小，乘回来才能保持两项量级相当。
'''

import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertConfig

# 复用 bert_train.py 里的数据读取 / 评估 / 报告函数，不重复造轮子
# （注意：bert_train 模块级会创建它自己的 config 实例，这里导入后即为同一个对象）
from bert_train import (
    config, load_raw_data, MyDataset, collate_fn, build_dataloaer,
    MyBertClassifier, evaluate, predict_all, report, train_one_epoch
)
from config import Config


# ============================ Student 模型 ============================
class StudentBertClassifier(nn.Module):
    """结构同 MyBertClassifier，但 Transformer 层数可配置。"""

    def __init__(self, num_layers=4):
        super().__init__()
        stu_cfg = BertConfig.from_pretrained(config.bert_path)
        stu_cfg.num_hidden_layers = num_layers
        # 这里不用 from_pretrained，先用配置随机建一个空壳，权重随后从 Teacher 拷贝
        self.bert = BertModel(stu_cfg)
        self.fc = nn.Linear(stu_cfg.hidden_size, len(config.id2class))

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(output.pooler_output)


def pick_layer_indices(total_layers, keep_layers, strategy='uniform'):
    """决定 Student 的每一层，分别拷贝 Teacher 的哪一层。"""
    if strategy == 'first':
        return list(range(keep_layers))
    if strategy == 'last':
        return list(range(total_layers - keep_layers, total_layers))
    # uniform：等间隔取样，np.linspace 的等价实现
    return [round(i * (total_layers - 1) / (keep_layers - 1)) for i in range(keep_layers)]


def init_student_from_teacher(student, teacher, num_layers, strategy='uniform'):
    """把 Teacher 的权重搬到 Student：embedding 直接用，Transformer 层按索引挑。"""
    teacher_layers = teacher.bert.encoder.layer
    total = len(teacher_layers)
    idx = pick_layer_indices(total, num_layers, strategy)
    print(f'Teacher 层数 {total} -> Student 层数 {num_layers}（策略 {strategy}）')
    print(f'Student 第 i 层分别拷贝 Teacher 的第 {idx} 层')

    # 1. Embedding（含 position / token_type）维度完全一致，直接整块搬
    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())

    # 2. 按索引拷贝 Transformer 层
    for i, src in enumerate(idx):
        student.bert.encoder.layer[i].load_state_dict(teacher_layers[src].state_dict())

    # 3. Pooler 与分类头维度一致，直接搬
    #    注意：Teacher 的 pooler 原本接在第 12 层之后，Student 接在第 4 层之后，
    #    严格说不完全匹配，但它只是一个 Linear+Tanh，微调几轮就能对齐。
    student.bert.pooler.load_state_dict(teacher.bert.pooler.state_dict())
    student.fc.load_state_dict(teacher.fc.state_dict())
    return student


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================ 蒸馏损失 ============================
def distill_loss_fn(student_logits, teacher_logits, labels, hard_loss_fn,
                    temperature=4.0, alpha=0.5):
    """
    硬损失 + 软损失（KL 散度）。

    KL 项必须用 log_softmax(student) 对 softmax(teacher)，
    且 teacher 的 logits 要 detach（Teacher 不参与更新）。
    """
    hard = hard_loss_fn(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return alpha * hard + (1 - alpha) * soft, hard, soft


# ============================ 蒸馏训练 ============================
def train_distill(student, teacher, train_dataloader, dev_dataloader, epochs=None):
    epochs = epochs or config.distill_epochs
    T = config.distill_temperature
    alpha = config.distill_alpha

    # 硬损失部分沿用类别加权（欺诈是少数类，不加权会拉低召回）
    if getattr(config, 'use_class_weight', False):
        lb = torch.tensor([int(x[1]) for x in train_dataloader.dataset.data_list])
        counts = torch.bincount(lb, minlength=len(config.id2class)).float()
        weights = (lb.numel() / (len(counts) * counts)).sqrt()
        print('类别权重:', {config.id2class[i]: round(w.item(), 3) for i, w in enumerate(weights)})
        hard_loss_fn = nn.CrossEntropyLoss(weight=weights.to(config.device))
    else:
        hard_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)

    teacher.eval()          # Teacher 全程只做前向，不更新、不开 dropout
    best_f1, best_state = 0.0, None
    history = {'hard': [], 'soft': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(1, epochs + 1):
        student.train()
        total, n_batch = 0.0, 0
        sum_hard, sum_soft = 0.0, 0.0
        loop = tqdm(train_dataloader, desc=f'Distill epoch {epoch}/{epochs}')
        for input_ids, attention_mask, labels in loop:
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            # Teacher 前向：不建图、不算梯度
            with torch.no_grad():
                teacher_logits = teacher(input_ids, attention_mask)

            student_logits = student(input_ids, attention_mask)
            loss, hard, soft = distill_loss_fn(
                student_logits, teacher_logits, labels, hard_loss_fn, T, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            sum_hard += hard.item()
            sum_soft += soft.item()
            n_batch += 1
            loop.set_postfix(loss=round(total / n_batch, 4))

        # 每轮结束在验证集上评估
        val_loss, val_acc, val_f1 = evaluate(student, dev_dataloader, hard_loss_fn)
        history['hard'].append(sum_hard / n_batch)
        history['soft'].append(sum_soft / n_batch)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        flag = ''
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, config.bert_student_model_path)
            flag = '  <== 刷新最优，已保存'
        print(f'epoch {epoch}: loss={total / n_batch:.4f} '
              f'(hard={sum_hard / n_batch:.4f}, soft={sum_soft / n_batch:.4f}) '
              f'val_acc={val_acc:.4f} val_f1={val_f1:.4f}{flag}')

    # 用验证集最优的那份权重做最终评估，不用最后一轮（可能已过拟合）
    if best_state is not None:
        student.load_state_dict(best_state)
    return student, history


# ============================ 主流程 ============================
def main():
    print('=' * 64)
    print('BERT 知识蒸馏：12 层 Teacher  ->  %d 层 Student' % config.student_layers)
    print('=' * 64)

    # ---------- 1. 数据 ----------
    train_data = load_raw_data(config.train_path)
    dev_data = load_raw_data(config.dev_path)
    test_data = load_raw_data(config.test_path)
    print(f'数据规模  训练 {len(train_data)} / 验证 {len(dev_data)} / 测试 {len(test_data)}')

    # build_dataloaer 内部已保证测试集 drop_last=False（不会漏样本）
    train_dl, dev_dl, test_dl = build_dataloaer(
        MyDataset(train_data), MyDataset(dev_data), MyDataset(test_data))

    # ---------- 2. Teacher ----------
    print('\n[1/4] 加载 Teacher（12 层微调 BERT）...')
    teacher = MyBertClassifier().to(config.device)
    teacher.load_state_dict(torch.load(config.bert_model_path, map_location=config.device))
    teacher.eval()
    # Teacher 在测试集上的成绩，作为本次蒸馏要追赶的标杆
    t0 = time.time()
    y_true, y_pred_t = predict_all(teacher, test_dl)
    teacher_time = (time.time() - t0) / len(test_data) * 1000
    report(y_true, y_pred_t, title='Teacher（12层 BERT）测试集')

    # ---------- 3. Student 初始化 ----------
    print('\n[2/4] 构建 Student 并从 Teacher 抽层初始化...')
    student = StudentBertClassifier(num_layers=config.student_layers).to(config.device)
    init_student_from_teacher(student, teacher, config.student_layers,
                              config.student_layer_strategy)

    tp, sp = count_params(teacher), count_params(student)
    print(f'\n参数量  Teacher {tp / 1e6:.1f}M  ->  Student {sp / 1e6:.1f}M'
          f'（压缩 {tp / sp:.2f}x）')

    # 蒸馏前先看一眼：只做了层裁剪、还没训练的小模型是什么水平
    y_t0, y_p0 = predict_all(student, test_dl)
    from sklearn.metrics import accuracy_score, f1_score
    acc0 = accuracy_score(y_t0, y_p0)
    f1_0 = f1_score(y_t0, y_p0, average='macro')
    print(f'抽层初始化后（尚未蒸馏）accuracy={acc0:.4f} macro-F1={f1_0:.4f}')

    # ---------- 4. 蒸馏训练 ----------
    print('\n[3/4] 开始蒸馏训练 '
          f'(T={config.distill_temperature}, alpha={config.distill_alpha}, '
          f'epochs={config.distill_epochs})...')
    student, history = train_distill(student, teacher, train_dl, dev_dl)

    # ---------- 5. 对比 ----------
    print('\n[4/4] 测试集最终对比')
    t0 = time.time()
    _, y_pred_s = predict_all(student, test_dl)
    student_time = (time.time() - t0) / len(test_data) * 1000
    report(y_true, y_pred_s, title=f'Student（{config.student_layers}层，蒸馏后）测试集')

    acc_t = accuracy_score(y_true, y_pred_t)
    f1_t = f1_score(y_true, y_pred_t, average='macro')
    acc_s = accuracy_score(y_true, y_pred_s)
    f1_s = f1_score(y_true, y_pred_s, average='macro')

    # 小模型和大模型预测一致的比例：蒸馏追求的就是"行为对齐"
    agree = sum(int(a) == int(b) for a, b in zip(y_pred_t, y_pred_s)) / len(y_pred_s)

    print('\n' + '=' * 64)
    print('蒸馏结果汇总')
    print('=' * 64)
    print(f'{"":<14}{"Teacher(12层)":>16}{"Student(%d层)" % config.student_layers:>16}{"变化":>14}')
    print(f'{"参数量(M)":<14}{tp / 1e6:>16.1f}{sp / 1e6:>16.1f}{"-%.1f%%" % ((1 - sp / tp) * 100):>14}')
    print(f'{"accuracy":<14}{acc_t:>16.4f}{acc_s:>16.4f}{acc_s - acc_t:>+14.4f}')
    print(f'{"macro-F1":<14}{f1_t:>16.4f}{f1_s:>16.4f}{f1_s - f1_t:>+14.4f}')
    print(f'{"单条耗时(ms)":<14}{teacher_time:>16.2f}{student_time:>16.2f}'
          f'{"%.2fx" % (teacher_time / student_time):>14}')
    print(f'{"与Teacher一致率":<14}{"":>16}{agree * 100:>15.1f}%{"":>14}')
    print(f'{"抽层初始化基线":<14}{"":>16}{f1_0:>16.4f}{"蒸馏提升 %+.4f" % (f1_s - f1_0):>20}')
    print('=' * 64)
    print(f'Student 已保存至: {config.bert_student_model_path}')


if __name__ == '__main__':
    main()
