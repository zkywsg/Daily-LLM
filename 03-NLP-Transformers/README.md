# Phase 03: NLP 与 Transformer

> 从"词是坐标"到"上下文决定含义"，再到"注意力就是一切"。
> 这一阶段回答了一个问题：如何让模型真正理解语言。

## 为什么这一阶段是分水岭

Phase 02 用 CNN + RNN 处理图像和序列。这一阶段的突破在于：

- **Word2Vec** 给词一个坐标，但每个词只有一个固定表示
- **Attention** 让解码器"回头看"编码器，打破了 RNN 的信息瓶颈
- **Transformer** 完全抛弃 RNN，用纯 Attention 处理序列，实现完全并行
- **BERT / GPT** 在大规模文本上预训练，一词多义被真正解决

读完这一阶段，你会理解：**为什么 Transformer 必然是大模型的基础架构**。

## 本阶段内容

### 1. [注意力机制](attention-mechanisms/README.md)
从 Bahdanau Attention 到 Self-Attention。
- Seq2Seq 的信息瓶颈问题
- Bahdanau / Luong Attention
- Self-Attention 的数学推导
- Multi-Head Attention 与位置编码

### 2. [Transformer 架构](transformer-architecture/README.md)
《Attention Is All You Need》的完整拆解。
- 编码器 / 解码器结构
- 残差连接 + Layer Normalization
- FFN 子层的作用
- 训练技巧：Warm-up、Label Smoothing

### 3. [预训练模型](pretrained-models/README.md)
BERT、GPT、T5 三条路线的系统对比。
- BERT：双向 Masked LM，下游任务微调
- GPT 系列：单向自回归，生成与上下文学习
- T5：统一文本到文本框架
- 主流变体：RoBERTa、ALBERT、DistilBERT、XLNet

## 时间线节点

本模块对应时间线中的以下关键工作：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2013 | Word2Vec | 词向量的起点，语义空间坐标系建立 |
| 2014 | GloVe | 全局共现矩阵词向量，与 Word2Vec 互补 |
| 2014 | Seq2Seq | 端到端机器翻译，Encoder-Decoder 范式确立 |
| 2014 | Bahdanau Attention | 信息瓶颈的第一个解法，Transformer 的直接前身 |
| 2016 | FastText | 子词级词向量，小语种和 OOV 问题的实用方案 |
| 2017 | Transformer | 纯 Attention 替代 RNN，完全并行，深度学习 NLP 新纪元 |
| 2018 | ELMo | 上下文动态词向量，一词多义的第一步 |
| 2018 | GPT-1 | Transformer 预训练 + 微调范式验证 |
| 2018 | BERT | 双向预训练，11 个 NLP 任务同时刷新 SOTA |
| 2018 | Transformer-XL | 片段级循环，突破固定上下文长度 |
| 2019 | GPT-2 | 1.5B 参数，生成质量震惊 NLP 社区 |
| 2019 | T5 | 统一文本到文本框架，任务大一统的早期体现 |
| 2019 | RoBERTa / XLNet / ALBERT | BERT 变体军备竞赛，预训练最佳实践确立 |
| 2019 | DistilBERT | 知识蒸馏压缩，轻量化 Transformer 的起点 |
| 2019 | HuggingFace Transformers | 开源 NLP 生态统一，成为事实标准 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)
