'''
    非结构化剪枝（Unstructured Pruning）

    ── 在干什么 ──────────────────────────────────────────────
    把 BERT 里那些"不重要"的权重直接置 0。这里用 L1 范数准则：
    单个权重绝对值越小，说明它对输出的贡献越小，越可以砍掉。

    ── 结构化 vs 非结构化 ─────────────────────────────────────
      · 结构化剪枝（structured）：整层 / 整个注意力头 / 整列砍掉
        -> 网络真的变窄，稠密矩阵乘法变少，通用硬件上就能加速。
      · 非结构化剪枝（unstructured）：单个权重随意置 0
        -> 得到的是稀疏矩阵，普通 CPU/GPU 依然按稠密算，
           除非用专门的稀疏算子（如 NVIDIA 2:4 sparsity、block-sparse kernel）。
    本脚本做的是后者，所以最后会看到：参数"少了一半"但推理速度几乎不变。
    这是剪枝中最容易踩的认知误区，脚本会把它量化出来。

    ── 为什么用渐进式（gradual pruning）─────────────────────────
    一次性剪掉 50% 权重，等于同时抽掉网络一半的"神经连接"，精度会断崖式下跌。
    改成"剪一点 -> 微调一轮 -> 再剪一点 -> 再微调"，
    让网络有时间把被剪掉的功能重新分配到保留下来的权重上，损失小得多。

    ── 全局剪枝 vs 逐层等比剪枝 ────────────────────────────────
    用 global_unstructured 而不是对每层单独设 amount：
    把所有 Linear 的权重放一起排序，谁小先剪谁。
    这样重要的层能自动保留更多参数，比"每层都剪 50%"这种一刀切更合理。
'''

import os
import io
import copy
import time
import gzip
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from bert_train import (
    config, load_raw_data, MyDataset, build_dataloaer,
    MyBertClassifier, evaluate, train_one_epoch, predict_all, report
)
from transformers import BertTokenizer

BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)

# 不参与剪枝的模块名：分类头和 pooler 参数量极小（768*6 / 768*768），
# 剪它们省不了什么空间，却直接损伤输出层表达力，性价比很低。
EXCLUDE_KEYWORDS = ('fc', 'pooler')


# ============================ 工具函数 ============================
def collect_linear_modules(model):
    """挑出要剪的 (module, 'weight') 组合，排除分类头与 pooler。"""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(k in name for k in EXCLUDE_KEYWORDS):
            targets.append((module, 'weight'))
    return targets


def global_sparsity(model):
    """统计当前模型真实的全局稀疏度：被置 0 的权重 / 参与剪枝的总权重。"""
    zero = total = 0
    for module, _ in collect_linear_modules(model):
        w = getattr(module, 'weight', None)
        if w is None:
            continue
        total += w.numel()
        zero += int((w == 0).sum().item())
    return zero / total if total else 0.0, zero, total


def save_compare(model, note=''):
    """同时给出：原始字节数、gzip 后字节数。

    为什么看 gzip 后的大小：非结构化剪枝只把权重置 0，
    用 float32 稠密格式存盘时，0 和别的数占一样的空间，文件大小不变。
    但稀疏矩阵有大量重复字节，gzip 能压得很小 —— 这才是"逻辑上省下的空间"。
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.tell()
    gz = len(gzip.compress(buf.getvalue(), compresslevel=6))
    return raw, gz


@torch.no_grad()
def latency_ms(model, sample_text, repeat=30, device=torch.device('cpu')):
    """CPU 上单条推理的平均耗时（毫秒）。"""
    model.eval().to(device)
    output = BERT_TOKENIZER(sample_text, max_length=config.max_len,
                            padding='max_length', truncation=True,
                            return_tensors='pt')
    input_ids = output['input_ids'].to(device)
    attention_mask = output['attention_mask'].to(device)
    for _ in range(5):          # 预热
        model(input_ids, attention_mask)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    for _ in range(repeat):
        model(input_ids, attention_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000


def eval_on_test(model, test_dl, device=None):
    """跑测试集。device 用于区分：剪枝模型在 GPU 上跑，量化模型只能在 CPU 跑。"""
    device = device or config.device
    old = config.device
    try:
        config.device = device            # predict_all 读的是这个模块级 config
        y_true, y_pred = predict_all(model, test_dl)
    finally:
        config.device = old
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


# ============================ 渐进式剪枝 ============================
def global_l1_mask(model, target_s):
    """
    全局 L1 幅度剪枝：算出使"累计稀疏度 = target_s"的阈值，返回 [(module, mask)]。

    为什么不用 torch.nn.utils.prune：
    prune.global_unstructured 会给每个 Linear 挂上 weight_orig + weight_mask
    两份 85M 的全精度缓冲，显存凭空多出约 0.68GB。本卡只有 4GB 且被浏览器等
    常驻程序分走一部分，实测会触发 WDDM 换页，训练速度从 6 it/s 掉到 0.05 it/s
    （20 秒一个 batch）。手动维护掩码可以让显存占用回到和未剪枝时一致。

    另外 PyTorch 的 amount 语义也容易踩坑：它是"在当前尚未被剪的权重中
    再剪掉的比例"，不是累计绝对比例。实测直接传 [1/6, 2/6, 3/6] 会复利叠加成
    16.67% -> 44.44% -> 72.22%，第三步远超计划的 50%。这里直接按
    "全局第 k 小的 |w|" 取阈值，天然就是累计语义，不存在换算问题。
    """
    targets = collect_linear_modules(model)

    # 阈值计算放 CPU：对 85M 个权重做全局 topk，4GB 卡上 GPU 版实测约 100s，
    # CPU 只要几秒。
    flats = torch.cat([m.weight.detach().abs().flatten().cpu() for m, _ in targets])
    total = flats.numel()
    k = int(round(target_s * total))
    if k <= 0:
        thr = -1.0
    elif k >= total:
        thr = float('inf')
    else:
        thr = torch.topk(flats, k, largest=False, sorted=True).values[-1].item()

    masks = []
    n_zero = 0
    for m, _ in targets:
        w = m.weight.data
        mask = (w.abs() > thr).to(w.device)   # 已置 0 的权重 |0| <= thr，天然被排除
        w.mul_(mask)                          # 原地置 0
        n_zero += int((~mask).sum().item())
        masks.append((m, mask))
    return masks, n_zero / total


@torch.no_grad()
def reapply_masks(masks):
    """每个 optimizer.step() 之后重新置 0。

    不这么做的话，梯度/Adam 动量/权重衰减会把被剪掉的权重重新"吹"回非零值，
    剪枝就白做了。这是手动掩码方案必须补的一步（PyTorch 的 prune 靠
    weight_mask 自动完成这件事）。
    """
    for m, mask in masks:
        m.weight.mul_(mask)


def finetune_one_epoch(model, train_dl, loss_fn, optimizer, masks, device=None):
    """带掩码重施加的微调轮。与 bert_train.train_one_epoch 等价，只多了 reapply。"""
    device = device or config.device
    model.train()
    total_loss, preds, gts = 0.0, [], []
    for input_ids, attention_mask, labels in tqdm(train_dl, desc='     微调'):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reapply_masks(masks)
        total_loss += loss.item()
        preds.extend(logits.argmax(-1).detach().cpu().tolist())
        gts.extend(labels.cpu().tolist())
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average='macro')
    return total_loss / max(len(train_dl), 1), acc, f1


def gradual_prune(model, train_dl, dev_dl, loss_fn, target_sparsity, steps,
                  finetune_epochs=1):
    """
    分 steps 步剪到 target_sparsity，每步剪完微调 finetune_epochs 轮。
    """
    targets = collect_linear_modules(model)
    n_prunable = sum(m.weight.numel() for m, _ in targets)
    print(f'参与剪枝的 Linear 层：{len(targets)} 个，共 {n_prunable / 1e6:.1f}M 个权重')

    schedule = [target_sparsity * (i + 1) / steps for i in range(steps)]
    print(f'剪枝计划（累计稀疏度）：{["%.0f%%" % (s * 100) for s in schedule]}')

    history = []
    dev = next(model.parameters()).device
    masks = []
    for step, target_s in enumerate(schedule, 1):
        masks, sp = global_l1_mask(model, target_s)
        zero = int(round(sp * n_prunable))
        print(f'\n--- 第 {step}/{steps} 步：已剪，当前稀疏度 {sp * 100:.2f}% '
              f'({zero / 1e6:.2f}M / {n_prunable / 1e6:.2f}M 个权重为 0) ---')

        # 剪完立刻微调，让网络自我修复。学习率减半，避免一下子把剩余权重改坏。
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr / 2,
                                      weight_decay=config.weight_decay)
        for ep in range(finetune_epochs):
            tr_loss, tr_acc, tr_f1 = finetune_one_epoch(
                model, train_dl, loss_fn, optimizer, masks, dev)
            val_loss, val_acc, val_f1 = evaluate(model, dev_dl, loss_fn)
            reapply_masks(masks)      # evaluate 不改权重，但保险起见
            print(f'    微调 epoch {ep + 1}: train_loss={tr_loss:.4f} '
                  f'val_acc={val_acc:.4f} val_f1={val_f1:.4f}', flush=True)
        history.append({'target': target_s, 'actual': sp,
                        'val_acc': val_acc, 'val_f1': val_f1})
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    # 手动掩码方案没有 weight_orig / weight_mask 缓冲，不需要 prune.remove 固化，
    # state_dict 天然干净，存盘体积也不会虚胖。
    sp, _, _ = global_sparsity(model)
    print(f'\n最终稀疏度 {sp * 100:.2f}%（无 prune 缓冲，可直接存盘）')
    return model, history


# ============================ 主流程 ============================
def main():
    print('=' * 66)
    print(f'BERT 非结构化剪枝  目标稀疏度 {config.prune_sparsity * 100:.0f}%  '
          f'分 {config.prune_steps} 步渐进')
    print('=' * 66)

    train_data = load_raw_data(config.train_path)
    dev_data = load_raw_data(config.dev_path)
    test_data = load_raw_data(config.test_path)
    train_dl, dev_dl, test_dl = build_dataloaer(
        MyDataset(train_data), MyDataset(dev_data), MyDataset(test_data))

    # ---------- 1. 基线（剪枝前原始模型）----------
    print('\n[1/5] 加载原始模型并测基线...')
    model = MyBertClassifier().to(config.device)
    model.load_state_dict(torch.load(config.bert_model_path, map_location=config.device))

    if getattr(config, 'use_class_weight', False):
        lb = torch.tensor([int(x[1]) for x in train_data])
        counts = torch.bincount(lb, minlength=len(config.id2class)).float()
        weights = (lb.numel() / (len(counts) * counts)).sqrt().to(config.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    base_acc, base_f1 = eval_on_test(model, test_dl)
    base_raw, base_gz = save_compare(model)
    # 延迟测量统一放 CPU，才和量化脚本的结论可比
    cpu_model = copy.deepcopy(model).to(torch.device('cpu'))
    base_lat = latency_ms(cpu_model, test_data[0][0])
    del cpu_model
    print(f'基线：accuracy={base_acc:.4f} macro-F1={base_f1:.4f} '
          f'体积={base_raw / 1024 ** 2:.1f}MB(gzip {base_gz / 1024 ** 2:.1f}MB) '
          f'CPU单条={base_lat:.2f}ms')

    # ---------- 2. 渐进剪枝 + 微调 ----------
    print('\n[2/5] 渐进式剪枝（每步剪完微调）...')
    model, history = gradual_prune(model, train_dl, dev_dl, loss_fn,
                                   config.prune_sparsity, config.prune_steps,
                                   config.prune_finetune_epochs)

    print('\n各稀疏度档位下的验证集表现（用于挑选"精度/稀疏度"平衡点）：')
    print(f'  {"目标稀疏度":<12}{"实际稀疏度":<12}{"val_acc":<10}{"val_f1":<10}')
    for h in history:
        print(f'  {h["target"] * 100:>8.0f}%   {h["actual"] * 100:>9.2f}%   '
              f'{h["val_acc"]:<10.4f}{h["val_f1"]:<10.4f}')

    # ---------- 3. 剪枝后评估 ----------
    print('\n[3/5] 剪枝后测试集评估')
    pruned_acc, pruned_f1 = eval_on_test(model, test_dl)
    pruned_raw, pruned_gz = save_compare(model)
    torch.save(model.state_dict(), config.bert_pruned_model_path)

    y_true, y_pred = predict_all(model, test_dl)
    report(y_true, y_pred, title=f'剪枝后（稀疏度 {config.prune_sparsity * 100:.0f}%）测试集')

    cpu_pruned = copy.deepcopy(model).to(torch.device('cpu'))
    pruned_lat = latency_ms(cpu_pruned, test_data[0][0])
    sp, zero, total = global_sparsity(cpu_pruned)

    # ---------- 4. 剪枝 + 量化叠加 ----------
    # 两者正交：剪枝决定"哪些连接没了"，量化决定"剩下的权重用几位表示"，可以叠加。
    print('\n[4/5] 剪枝 + 动态量化 叠加...')
    quant = torch.quantization.quantize_dynamic(
        copy.deepcopy(cpu_pruned), {nn.Linear}, dtype=torch.qint8)
    # 量化模型只能跑 CPU，这里显式指定设备，否则 predict_all 会把数据送到 cuda
    q_acc, q_f1 = eval_on_test(quant, test_dl, device=torch.device('cpu'))
    q_raw, q_gz = save_compare(quant)
    q_lat = latency_ms(quant, test_data[0][0])

    # ---------- 5. 汇总 ----------
    print('\n[5/5] 汇总')
    print('\n' + '=' * 74)
    print('非结构化剪枝结果汇总（测试集 %d 条）' % len(test_data))
    print('=' * 74)
    header = f'{"":<16}{"原始":>13}{"剪枝后":>13}{"剪枝+量化":>13}'
    print(header)
    print('-' * 74)
    print(f'{"accuracy":<16}{base_acc:>13.4f}{pruned_acc:>13.4f}{q_acc:>13.4f}')
    print(f'{"macro-F1":<16}{base_f1:>13.4f}{pruned_f1:>13.4f}{q_f1:>13.4f}')
    print(f'{"稀疏度":<16}{"0%":>13}{sp * 100:>12.1f}%{sp * 100:>12.1f}%')
    print(f'{"存盘体积(MB)":<16}{base_raw / 1024 ** 2:>13.1f}{pruned_raw / 1024 ** 2:>13.1f}'
          f'{q_raw / 1024 ** 2:>13.1f}')
    print(f'{"gzip后(MB)":<16}{base_gz / 1024 ** 2:>13.1f}{pruned_gz / 1024 ** 2:>13.1f}'
          f'{q_gz / 1024 ** 2:>13.1f}')
    print(f'{"CPU单条(ms)":<16}{base_lat:>13.2f}{pruned_lat:>13.2f}{q_lat:>13.2f}')
    print('-' * 74)
    print(f'零权重占比     : {sp * 100:.2f}%  ({zero / 1e6:.2f}M / {total / 1e6:.2f}M)')
    print(f'gzip 体积压缩  : {(1 - pruned_gz / base_gz) * 100:.1f}%'
          f'（剪枝+量化 {(1 - q_gz / base_gz) * 100:.1f}%）')
    print(f'CPU 推理加速   : {base_lat / pruned_lat:.2f}x'
          f'（剪枝+量化 {base_lat / q_lat:.2f}x）')
    print(f'精度代价       : accuracy {pruned_acc - base_acc:+.4f} / '
          f'macro-F1 {pruned_f1 - base_f1:+.4f}')
    print('=' * 74)
    print('\n⚠ 关键结论：非结构化剪枝只是把权重置 0，PyTorch 默认仍按稠密矩阵计算，')
    print(f'  所以稀疏度已达 {sp * 100:.0f}%，CPU 单条耗时却只从 {base_lat:.2f}ms '
          f'变成 {pruned_lat:.2f}ms —— 几乎没有加速。')
    print('  真正的收益在"体积"（gzip 后）和"与量化叠加"，')
    print('  想拿到推理加速必须换结构化剪枝（砍注意力头/层）或专用稀疏算子。')
    print(f'\n模型已保存至: {config.bert_pruned_model_path}')


if __name__ == '__main__':
    main()
