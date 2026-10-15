---
tasks:
- fill-mask
widgets:
  - task: fill-mask
    inputs:
      - type: text
        validator:
          max_words: 500
    examples:
      - name: 1
        inputs:
          - data: 中国的首都是[MASK]京
      - name: 2
        inputs:
          - data: 法国的首都是[MASK]黎
      - name: 3
        inputs:
          - data: 今天天气不好，可能会下[MASK]？
domain:
- nlp
frameworks:
- PyTorch
model-type:
- bert
backbone:
- transformer
metrics:
- F1
language:
- cn
license: Apache License 2.0
tags:
- AI日日新
- transformers
- 预训练
- 完形填空

finetune-support: True

---

## 微调该模型
请见魔搭社区仓库[python-data-science](https://www.modelscope.cn/models/tiansz/python-data-science/summary)


## 克隆该模型
注意克隆前请下载 [git](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git) 和 [git lfs](https://git-lfs.com/)
```bash
 git clone https://www.modelscope.cn/tiansz/bert-base-chinese.git
```

当然，你也可以使用modelscope的命令行工具下载本模型，请先安装modelscope：使用`pip install modelscope` 或`uv add modelscope`，然后运行如下命令：
```bash
modelscope download --model tiansz/bert-base-chinese --local_dir <your_loacl_path>
```

## 来源

该模型是 huggingface 的 bert-base-chinese 的模型备份，该模型可以用于实现文本分类、实体识别等任务。

它是一个基于Transformer架构的中文预训练模型，使用了大量的中文语料进行训练。它在多个中文自然语言处理任务上表现出色，如文本分类、命名实体识别和情感分析等。但我们还有效果更好的中文模型可选择，例如哈工大与讯飞的[lert](https://www.modelscope.cn/models/pengzhendong/chinese-lert-base/summary)模型等

通过使用bert-base-chinese，我们可以将其用作下游任务的特征提取器或者进行微调以适应特定任务。对应的各下游微调notebook可见[github仓库](https://github.com/tiansztiansz/python-data-science/tree/main)，若你无法打开github网站，可以下载[watt toolkit](https://steampp.net/)帮你解决该问题

















