# 广告素材合规审核系统

基于中文预训练模型的广告文案合规审核方案，覆盖 **随机森林 / fastText / BERT 微调 / 大语言模型** 四条技术路线，并完整实现了 **置信度拒判、量化、知识蒸馏、非结构化剪枝** 四套工程化手段。

核心思路：把「一段广告文案 → 一个合规类别」建模为单标签文本分类问题，同时对低置信度样本自动转人工复核，避免模型在拿不准的case上强行输出。

---

## 一、业务背景

广告投放前需要对素材做合规审核，传统人工全量审核成本高、标准不一致。本项目提供机器初筛 + 置信度分流的方案：

- **机器可判**（置信度 ≥ 阈值）→ 自动通过或拦截
- **机器无把握**（置信度 < 阈值）→ 转人工复核

按当前实验，**约 96% 的样本可自动化处理**，人工产能放大约 25 倍。

## 二、数据集

来源：[zjunlp/ChineseHarm-bench](https://huggingface.co/datasets/zjunlp/ChineseHarm-bench)（HuggingFace）

去重后共 **9503 条**，划分为 train 7600 / dev 947 / test 956。标签体系（`01-data/class.txt`）顺序如下：

| 序号 | 类别 | 说明 |
|:---:|------|------|
| 0 | 不违规 | 正常广告文案 |
| 1 | 黑产广告 | 灰产/黑产推广、代办证件等 |
| 2 | 欺诈 | 虚假宣传、诈骗引流 |
| 3 | 博彩 | 赌博引流、私彩推广 |
| 4 | 谩骂引战 | 攻击性言论、挑拨对立 |
| 5 | 低俗色情 | 软色情、擦边内容 |

数据格式约定为 `文本\t标签`（LF 行尾），转换脚本见 `01-data/prepare_adaudit_data.py`。

## 三、实验结果

### 3.1 四方案横向对比（测试集）

| 方案 | 准确率 | Macro-F1 | 特点 |
|------|:---:|:---:|------|
| 随机森林 (TF-IDF) | 0.78 | — | 基线，训练快，可解释 |
| fastText（字级） | 0.908 | — | 速度与精度平衡好 |
| fastText（词级） | 0.872 | — | 分词质量影响明显 |
| fastText（自动调参） | 0.859–0.911 | — | 注意 dim 过大易 OOM |
| **BERT 微调（最终）** | **0.9331** | **0.934** | 加入类别权重后欺诈召回 0.77→0.94 |

BERT 加权前后的关键收益：**欺诈类召回率从 0.77 提升到 0.94**。原因是欺诈类在数据中占比小，加权前的模型倾向于牺牲少数类换整体准确率——而合规场景里漏判欺诈的代价远高于误判。

### 3.2 模型压缩（BERT 路线）

| 手段 | 体积 / 参数量 | 推理延迟 | 准确率 | Macro-F1 |
|------|:---:|:---:|:---:|:---:|
| 原始模型 | 391MB / 102.3M | 5.19ms | 0.9331 | 0.934 |
| 动态量化 | 390MB → **146MB** | 1.41× 加速 | -1.8 pt | — |
| 知识蒸馏（12→4层） | 102.3M → **45.6M**（2.24×） | 5.19ms → **1.68ms**（3.08×） | 0.9247 | 0.9230 |
| 非结构化剪枝 50% | 稀疏度 50% | — | 0.9320 | 0.9305 |
| 剪枝 + 量化 | — | — | 0.9163 | 0.9146 |

蒸馏的一个关键教训：`alpha=0.5` 时 KL 损失不带类别权重，会把加权换来的欺诈召回从 Teacher 的 0.94 拉回 Student 的 0.77。改为 **alpha=0.7 + 5 epochs** 后，Student 欺诈召回恢复到 0.83，一致率 92.6%。

另外请注意：**非结构化剪枝省的是压缩后的体积（gzip 后 244.3MB），不是磁盘占用，也不带来 CPU 加速**——稀疏矩阵需要专用硬件/算子才能提速。

## 四、目录结构

```
code/
├── 01-data/              数据处理
│   ├── prepare_adaudit_data.py    原始数据 → train/dev/test.txt
│   ├── data_eda.py                数据分布探索
│   ├── _raw/                      HuggingFace 下载的原始 json
│   ├── train/dev/test.txt         BERT 路线数据（字级）
│   ├── char_*.txt / word_*.txt    fastText 路线数据
│   └── bert-base-chinese/         预训练模型（权重需另行下载）
├── 02-rf/                随机森林方案：训练 / 推理 / Streamlit
├── 03-fasttext/          fastText 方案：含自动调参变体
├── 04_Bett/              BERT 方案（主路线）
│   ├── bert_train.py              训练（类别加权、混淆矩阵、曲线）
│   ├── bert_predict_fun.py        推理（含置信度拒判）
│   ├── bert_quantization.py       动态量化
│   ├── bert_distillation.py       知识蒸馏（Teacher 12层 → Student 4层）
│   ├── bert_pruning.py            非结构化剪枝（全局 L1）
│   ├── config.py                  集中式配置
│   └── 实验报告.html/pdf          完整实验报告
├── 05-LLM/               大模型方案：DeepSeek API + 提示词工程
├── 00-test/ / test/      Flask 入门练习代码
└── 项目文件说明.html/pdf   逐文件功能说明文档
```

## 五、快速开始

### 5.1 环境

```bash
conda create -n DeepLearning python=3.10
conda activate DeepLearning
pip install torch transformers scikit-learn fasttext pandas matplotlib seaborn flask streamlit
```

> fastText 的安装容易在 Windows 上踩坑，建议直接用 `pip install fasttext-wheel`。

### 5.2 下载预训练模型

本仓库**不包含模型权重**（单个文件超过 GitHub 100MB 上限）。请自行下载：

```bash
huggingface-cli download bert-base-chinese model.safetensors \
  --local-dir 01-data/bert-base-chinese
```

或者用 ModelScope 镜像（国内更快）：

```bash
pip install modelscope
modelscope download --model AI-ModelScope/bert-base-chinese \
  --local_dir 01-data/bert-base-chinese
```

> 词表文件（`vocab.txt`、`tokenizer.json`、`config.json`）已在仓库中，无需重复下载。

### 5.3 训练与推理

```bash
cd 04_Bett
python bert_train.py              # 训练，权重保存到 ../01-data/save_model/
python bert_predict_fun.py        # 推理
python bert_quantization.py       # 量化
python bert_distillation.py       # 蒸馏
python bert_pruning.py            # 剪枝
streamlit run streamlit_app.py    # 可视化 Demo
```

### 5.4 LLM 方案配置

`05-LLM/` 需要一个 `.env` 文件（**该文件已被 gitignore，请自行创建**）：

```
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

## 六、已知坑与实践经验

| 问题 | 现象 | 解决方式 |
|------|------|---------|
| tqdm 进度条卡死 | `python train.py \| grep` 时因 `\r` 不换行攒成巨型单行，曾卡死 16 分钟 | 一律重定向到 `*.log`，再用 `tr '\r' '\n'` 查看 |
| 显存换页 | RTX 3050Ti 4GB 下 batch_size=16 触发 WDDM 换页，掉到 <1 it/s | 降到 batch_size=8，约 6 it/s |
| fastText 打不开中文路径 | C++ 层不支持窄字符路径 | config 里加 `FT_ASCII_DIR`，复制到 `%TEMP%` 下的 ASCII 目录运行 |
| 量化时 device 撞车 | 数据送 GPU、模型在 CPU | 量化脚本与 `bert_train` 模块的 config 都强制 `cpu` |
| 剪枝稀疏度不准 | `prune.global_unstructured` 的 amount 是相对「剩余未剪权重」，直接传累计值会复利到 72% | 改用 `bert_pruning.py` 中手写的 `global_l1_mask` |
| 端口冲突 | RF / fastText / BERT 的 `api_port` 默认都是 6000 | 同时启动多个服务前手动改 port |

## 七、局限说明

这个项目是教学练手性质，**不能直接上生产**，理由如下：

1. **先验分布不同**：真实投放场景中违规素材占比通常远低于实验集的分布，命中率（precision）会明显下降，阈值需要按实际 prior 重新标定。
2. **仅覆盖文本**：实际广告素材包含图片和视频，图文不一致是常见的违规手法。
3. **标签体系不全**：六类无法覆盖全部违规形态（如医疗器械夸大疗效、金融承诺保本等专项条款）。
4. **黑话漂移**：黑产话术迭代很快，模型需要持续用新样本运营更新。

## 八、免责声明

本项目使用的数据集来自公开学术研究资源，模型输出仅供参考，不构成法律意见，也不应作为唯一的合规判定依据。实际投放请以监管部门要求和法务审核为准。
