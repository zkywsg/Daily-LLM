# Phase 02 · 语言线（2013–2019）

从词是坐标，到上下文决定含义，再到注意力就是一切。
这条线在 2021 年 CLIP 处与视觉线汇流。

## 本阶段内容

### [循环神经网络与 Seq2Seq](recurrent-networks/README.md)
RNN、LSTM、GRU，到编码器-解码器翻译范式
- 固定窗口方法为什么不够用
- 循环状态、BPTT 与长依赖难题
- LSTM / GRU 的门控机制
- Seq2Seq 如何推动 Attention 出现

### [注意力机制](attention-mechanisms/README.md)
从 Bahdanau Attention 到 Self-Attention
- Seq2Seq 的信息瓶颈问题
- Bahdanau / Luong Attention 数学推导
- Self-Attention 与 Multi-Head Attention
- 位置编码

### [Transformer 架构](transformer-architecture/README.md)
《Attention Is All You Need》完整拆解
- Encoder / Decoder 结构
- 残差连接 + Layer Normalization
- 训练技巧：Warm-up、Label Smoothing

### [预训练模型](pretrained-models/README.md)
BERT、GPT、T5 三条路线系统对比
- ELMo：上下文动态词向量
- GPT-1/2：单向自回归预训练
- BERT：双向 Masked LM
- T5：统一文本到文本框架
- 变体：RoBERTa、ALBERT、DistilBERT、XLNet

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2013 | Word2Vec | 词向量起点，语义空间坐标系 |
| 2014 | GloVe / Seq2Seq / Attention | 全局共现词向量；端到端翻译；信息瓶颈的第一个解法 |
| 2016 | FastText | 子词级词向量，OOV 问题解决 |
| 2017 | Transformer | 纯 Attention 取代 RNN，完全并行 |
| 2018 | ELMo / GPT-1 / BERT | 动态词向量；预训练 + 微调范式确立 |
| 2019 | GPT-2 / T5 / RoBERTa | 规模化生成；任务统一；BERT 训练不足被证明 |
| 2019 | ALBERT / DistilBERT | 参数共享压缩；知识蒸馏轻量化 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [汇流：规模与多模态 →](../03-Scale-Multimodal/)
