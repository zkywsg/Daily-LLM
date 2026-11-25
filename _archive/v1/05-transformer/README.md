# Transformer 架构

> **把 attention 推到极致,扔掉循环和卷积,让序列建模获得完全并行 + 全局上下文。**

## 一句话定位

Transformer 家族解决的是序列建模里两个一直没拿掉的硬伤:**循环结构的串行 + 长上下文的代价**。[RNN](../02-rnn-lstm/01-rnn.md) 在时间上一步一步展开,无法并行;CNN 用卷积换并行但感受野受局部窗口限制;[Bahdanau Attention](../02-rnn-lstm/05-attention.md) 给 Seq2Seq 装上"按需回看"但仍跑在循环骨架上。2017 年 Vaswani 等人的 *Attention Is All You Need* 给出的解法是结构性的——**把循环骨架完全扔掉,只用 attention 做信息聚合**——这让训练时序列上的每个位置都可以并行计算,同时每个位置一步就能看到所有其他位置。这一架构在 2017 年发表时只是机器翻译领域的 SOTA,但接下来 5 年里它几乎吞掉了整个深度学习——NLP 的 [BERT](../06-bert-family/) / [GPT](../07-gpt-scaling/)、视觉的 [ViT](../08-vit/)、跨模态的 [CLIP](../09-multimodal-clip/) / [Diffusion](../10-diffusion/) 全部以 Transformer 为基座。这家族要回答的问题是:**从 2017 原版到 2023 LLaMA 时代,Transformer 架构本身经历了哪些演化**。

## 概念本身

Transformer 的核心机制是 **scaled dot-product attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

把 Bahdanau 的"目标位置对源序列学一个加权分布"推到极致——**让序列里每个位置同时对所有其他位置做加权聚合**,权重由 query-key 内积决定。这一操作有三个关键性质:

1. **完全并行**——所有位置的 attention 一次矩阵乘算完,GPU 友好。RNN/CNN 的局部展开换成了一次性全局 matmul
2. **全局上下文**——任意两个位置的"距离"都是 O(1) 步,长依赖不再需要梯度沿时间链回传
3. **置换等变**——attention 本身不区分位置先后,需要外加 **position encoding** 把序列顺序注入回来

把这一基本块堆叠 N 次,再配上 **multi-head**(让不同 head 学不同关系)、**FFN**(每个位置独立的 2 层 MLP)、**残差 + LayerNorm**(深度训练的稳定剂),就得到了 Transformer 的 encoder/decoder。这套设计在 2017 年看起来像是"Bahdanau attention + Seq2Seq 的极端化",但事后看,它做对的几件事都是结构性的——并行化、全局聚合、模块化堆叠——这让它能从 6 层 65M 参数的翻译模型扩展到 96 层 175B 参数的 GPT-3,**架构本体几乎不需要改**。

这家族的演化主线围绕几个明确的瓶颈:

- **复杂度**:原版 attention 是 O(N²),长上下文(>4K)代价太高。Sparse / Linear / FlashAttention 各自从算法和系统层面攻这件事
- **位置编码**:原版正余弦 PE 在长上下文下泛化差,RoPE / ALiBi 给出更鲁棒的方案
- **训练稳定性**:Post-LN 在深层时梯度不稳,Pre-LN / RMSNorm 成为深度模型默认
- **推理效率**:多头 attention 在 KV cache 时代变成内存瓶颈,MQA / GQA 通过共享 KV 头压缩 cache
- **长上下文**:从 2017 的 512 token 到 2024 的 1M+ token,长上下文是过去 5 年最持续的方向

理解这一家族不只是为了用 Transformer,而是为了理解**现代 LLM 的架构基础是怎么搭起来的**——一个 LLaMA 2 模型用的是 Pre-RMSNorm + RoPE + SwiGLU + GQA,这些每一个都是 2017 之后才出现的。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2017 | **Transformer** | 用 self-attention 替代循环,encoder-decoder 骨架完全并行化,multi-head + scaled dot-product + 正余弦 PE 一起定型 | RNN/LSTM 串行训练慢,长序列翻译质量下降;CNN 序列模型感受野受局部窗口限制 |
| 2019 | **Transformer-XL** | segment-level recurrence + 相对位置编码,把上下文长度从固定窗口推到可累积的"片段记忆" | 原版 PE 是绝对位置,跨片段无法泛化;固定上下文截断破坏长依赖 |
| 2020 | **Sparse Attention** | Longformer/BigBird 用滑窗 + 全局 token 把 attention 复杂度从 O(N²) 降到 O(N),支持 4K–16K 长上下文 | 原版 attention O(N²),GPU 内存随长度平方增长,4K 以上不可行 |
| 2021 | **RoPE** | 旋转位置编码,把位置信息编码进 Q/K 的内积里,长上下文外推更鲁棒,成为 LLaMA/GPT 系标配 | 正余弦 PE 在训练长度之外泛化差;learned PE 完全不外推 |
| 2022 | **FlashAttention** | IO-aware attention 实现,通过分块 + 重计算把 attention 的 HBM 访存压到最低,训练快 2-4× 且支持更长序列 | attention 的瓶颈不是 FLOPs 而是 GPU 内存带宽;naive 实现要把 N×N attention 矩阵物化到 HBM |

## 依赖与延伸

**前置(foundations):**
- `../foundations/04-normalization/` —— LayerNorm / RMSNorm 是深层 Transformer 的稳定剂
- `../foundations/02-activations/` —— GELU / SwiGLU 在 FFN 里的应用
- `../foundations/05-initialization/` —— 深层 Transformer 的初始化和 learning rate warmup
- [../02-rnn-lstm/05-attention.md](../02-rnn-lstm/05-attention.md) —— Bahdanau attention 是 Transformer cross-attention 的直系前作

**通向哪些家族:**
- [../06-bert-family/](../06-bert-family/) —— encoder-only Transformer + masked LM 预训练
- [../07-gpt-scaling/](../07-gpt-scaling/) —— decoder-only Transformer + 自回归生成 + scaling laws
- [../08-vit/](../08-vit/) —— 把 Transformer 直接用到视觉,patch as token
- [../09-multimodal-clip/](../09-multimodal-clip/) —— 跨模态对齐基于 Transformer 双塔
- [../10-diffusion/](../10-diffusion/) —— DiT(Diffusion Transformer)用 Transformer 替代 U-Net
- [../13-moe-efficient/](../13-moe-efficient/) —— MoE / 量化 / 蒸馏,Transformer 高效化的工程路线
