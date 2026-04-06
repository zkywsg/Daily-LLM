# Phase 02 · 语言线（2013–2019）

从词是坐标，到上下文决定含义，再到注意力和预训练把语言建模变成基础设施。
这条线在 2021 年 CLIP 处与视觉线汇流。

## 这个问题从哪来
> 2013 到 2019 年，语言建模从“只看局部窗口”走向“看完整上下文再生成”。
> Word2Vec、Seq2Seq、Attention、Transformer 和预训练，依次解决了表示、记忆、瓶颈与迁移问题。

## 学习目标
完成后你应能回答：1. 为什么固定窗口和纯循环都不够用  2. Attention 如何缓解信息瓶颈  3. 为什么预训练把 NLP 从任务模型变成基础模型

## 1. 直觉
自然语言不是一张局部纹理稳定的图，它更像一段需要连续跟踪的对话。
RNN 先把“历史”压进状态里，但状态会越来越难记住早期信息。
Attention 让模型每一步都能回头看关键位置，不必把所有信息硬塞进一个向量。
预训练则进一步把“先学语言，再做任务”变成通用做法，语言模型开始像底座而不是单个工具。

## 2. 机制
这条线的机制其实是四次递进：
1. RNN / Seq2Seq 解决“怎么按顺序建模”。
2. Attention 解决“怎么按需回看上下文”。
3. Transformer 解决“怎么去掉循环、提升并行”。
4. Pretraining 解决“怎么把语言能力复用到更多任务”。

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

→ 完整时间线见 [00-Timeline](../00-Timeline/README.md)

## 3. 工程陷阱
- 固定窗口太短：只能看到局部上下文，长程依赖直接丢失。
- 循环状态太脆：早期信息在 BPTT 中越来越难传回去，训练也更慢。
- 单向编码太窄：只做左到右生成时，理解任务会损失双向上下文。
- 预训练与微调有落差：模型学到的是通用表征，落到任务上还需要对齐数据分布和目标函数。

## 演进笔记
> 这一阶段的遗产：语言建模从序列技巧变成可复用的通用表征；遗留的新问题是规模、对齐和多模态。
→ 详见 [03-Scale-Multimodal](../03-Scale-Multimodal/)

**下一阶段**: [汇流：规模与多模态 →](../03-Scale-Multimodal/)
