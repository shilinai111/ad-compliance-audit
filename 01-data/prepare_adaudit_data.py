"""
广告合规审核数据集准备脚本
    数据源：HuggingFace  zjunlp/ChineseHarm-bench（中文有害内容检测基准）
    业务场景：广告素材上线前的合规审核 / 违规内容拦截

作用
    把 _raw/chineseharm/*.json 转换成项目统一的数据契约：
        train.txt / dev.txt / test.txt   每行格式：  文本\t标签索引
        class.txt                        每行一个类别名，行号即标签 ID

只需跑一次。之后 02-rf 和 03-fasttext 各自的 data_process.py
会自动从这三个文件派生出 char_*.txt / word_*.txt / process_*.csv。

用法
    python 01-data/prepare_adaudit_data.py
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- 1. 配置

# 类别顺序 = 标签索引。顺序一旦确定就不要再改，
# 否则之前训练好的模型权重会对不上类别。
CLASS_LIST = [
    "不违规",      # 0  正常内容，允许投放
    "黑产广告",    # 1  灰产/黑产推广，业务核心打击对象
    "欺诈",        # 2
    "博彩",        # 3
    "谩骂引战",    # 4
    "低俗色情",    # 5
]

ROOT_DIR = Path(__file__).parent
RAW_DIR = ROOT_DIR / "_raw" / "chineseharm"

TRAIN_PATH = ROOT_DIR / "train.txt"
DEV_PATH = ROOT_DIR / "dev.txt"
TEST_PATH = ROOT_DIR / "test.txt"
CLASS_PATH = ROOT_DIR / "class.txt"

TRAIN_RATIO, DEV_RATIO = 0.8, 0.1   # 剩下的是 test，即 8 : 1 : 1
RANDOM_SEED = 42
MIN_LEN = 2                          # 过滤掉过短的噪声样本


# ---------------------------------------------------------------- 2. 清洗
# 数据契约用 \t 分隔文本和标签，所以文本里绝对不能出现 \t；
# 换行同样要处理，否则会破坏"一行一条样本"的约定。
_WS_RE = re.compile(r"[\t\r\n]+")


def clean_text(text: str) -> str:
    text = _WS_RE.sub(" ", str(text)).strip()
    return text


# ---------------------------------------------------------------- 3. 读取
def load_all() -> list[tuple[str, str]]:
    """读取所有分类 json，返回 [(文本, 类别名)]"""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到原始数据目录：{RAW_DIR}\n"
            f"请先把 zjunlp/ChineseHarm-bench 的 json 放到该目录下。"
        )

    samples: list[tuple[str, str]] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n = 0
        for item in data:
            text = clean_text(item.get("文本", ""))
            label = str(item.get("标签", "")).strip()
            if len(text) < MIN_LEN or label not in CLASS_LIST:
                continue
            samples.append((text, label))
            n += 1
        print(f"  读取 {path.name:<16} 有效样本 {n:>6}")

    return samples


# ---------------------------------------------------------------- 4. 去重
def dedup(samples: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int, int]:
    """
    去重策略：
      - 完全相同的 (文本, 标签) 只留一条
      - 同一文本被标成不同类别的（标签冲突）直接丢弃，
        这类样本是标注噪声，留着会污染训练和评估
    """
    by_text: dict[str, set[str]] = {}
    for text, label in samples:
        by_text.setdefault(text, set()).add(label)

    clean = [(t, next(iter(labels))) for t, labels in by_text.items()
             if len(labels) == 1]
    conflict = sum(1 for labels in by_text.values() if len(labels) > 1)
    return clean, len(samples) - len(by_text), conflict


# ---------------------------------------------------------------- 5. 切分
def split(samples: list[tuple[str, str]]):
    """按类别分层抽样，保证三份数据集里各类别比例一致"""
    random.seed(RANDOM_SEED)

    buckets: dict[str, list[tuple[str, str]]] = {c: [] for c in CLASS_LIST}
    for text, label in samples:
        buckets[label].append((text, label))

    train, dev, test = [], [], []
    for label, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_dev = int(n * DEV_RATIO)
        train += items[:n_train]
        dev += items[n_train:n_train + n_dev]
        test += items[n_train + n_dev:]

    for part in (train, dev, test):
        random.shuffle(part)
    return train, dev, test


# ---------------------------------------------------------------- 6. 落盘
def dump(part: list[tuple[str, str]], path: Path):
    idx = {name: i for i, name in enumerate(CLASS_LIST)}
    # newline='\n'：Windows 下默认会把 \n 变成 \r\n，
    # 下游 split('\t') 取到的标签可能带着 \r，导致 int() 报错。
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for text, label in part:
            f.write(f"{text}\t{idx[label]}\n")


def main():
    print("=" * 60)
    print("广告合规审核 · 数据准备")
    print("=" * 60)

    print("\n[1/5] 读取原始数据")
    samples = load_all()
    print(f"  合计 {len(samples)} 条")

    print("\n[2/5] 去重")
    samples, dup, conflict = dedup(samples)
    print(f"  重复样本 {dup} 条，标签冲突丢弃 {conflict} 条，剩余 {len(samples)} 条")

    print("\n[3/5] 分层切分 8:1:1")
    train, dev, test = split(samples)
    print(f"  train {len(train)} / dev {len(dev)} / test {len(test)}")

    print("\n[4/5] 写入文件")
    dump(train, TRAIN_PATH)
    dump(dev, DEV_PATH)
    dump(test, TEST_PATH)
    with open(CLASS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(CLASS_LIST) + "\n")

    for p in (TRAIN_PATH, DEV_PATH, TEST_PATH, CLASS_PATH):
        print(f"  {p.name:<12} {os.path.getsize(p):>9,} 字节")

    print("\n[5/5] 类别分布")
    idx = {name: i for i, name in enumerate(CLASS_LIST)}
    print(f"  {'类别':<10}{'索引':<6}{'train':>8}{'dev':>7}{'test':>7}")
    for name in CLASS_LIST:
        c_train = sum(1 for _, l in train if l == name)
        c_dev = sum(1 for _, l in dev if l == name)
        c_test = sum(1 for _, l in test if l == name)
        print(f"  {name:<10}{idx[name]:<6}{c_train:>8}{c_dev:>7}{c_test:>7}")

    print("\n完成。接下来可以跑：")
    print("  python 04_Bett/bert_train.py")


if __name__ == "__main__":
    main()
