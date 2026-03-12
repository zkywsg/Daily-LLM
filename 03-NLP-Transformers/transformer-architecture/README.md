[English](README_EN.md) | [中文](README.md)

# Transformer 架构

## 概述

Transformer 用“注意力 + 前馈网络 + 残差归一化”取代了 RNN 的时序递归，显著提升并行训练能力。理解 Transformer 的关键不是背结构图，而是看懂一个 block 如何在信息混合与稳定训练之间平衡。

## 学习目标

完成本章后，你应能回答：

1. 一个 Transformer block 的最小组成和数据流是什么？
2. Encoder-only、Decoder-only、Encoder-Decoder 的任务边界如何区分？
3. 训练和推理时最常见的稳定性与效率问题是什么？

## 1. 高层结构一图理解

```text
Token Embedding + Position
        ↓
   [Transformer Block] × N
        ↓
Task Head (分类/生成/序列到序列)
```

对于 Seq2Seq（翻译、摘要）：

```text
源序列 -> Encoder 堆栈
目标序列 -> Decoder 堆栈 (含因果自注意力 + 对 Encoder 的交叉注意力)
```

你要记住：Transformer 是“重复堆叠同一计算单元”的架构家族。

## 2. 一个 Transformer Block

标准 block 包含两层子结构：

1. Multi-Head Attention（跨 token 信息交互）
2. Feed-Forward Network（逐 token 非线性变换）

并在每个子层外包裹：

- Residual Connection（保梯度）
- LayerNorm（稳训练）

常见写法是 Pre-LN：

$$
x = x + \text{MHA}(\text{LN}(x)),\quad
x = x + \text{FFN}(\text{LN}(x))
$$

你要记住：注意力负责“token 间通信”，FFN 负责“token 内变换”。

## 3. 编码器层与解码器层（PyTorch）

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x
```

解码器层相比编码器多一个交叉注意力：

1. 因果自注意力（不能看未来 token）
2. 交叉注意力（读 encoder 输出）
3. FFN

## 4. 三种主流范式与应用边界

| 范式 | 结构 | 典型任务 | 代表模型 |
|------|------|----------|----------|
| Encoder-only | 仅编码器 | 分类、检索、序列标注 | BERT/RoBERTa |
| Decoder-only | 仅解码器 | 自回归生成、对话 | GPT 系列 |
| Encoder-Decoder | 编解码器 | 翻译、摘要、改写 | T5/BART |

你要记住：任务是“理解”还是“生成”，决定了你该选哪类 Transformer。

## 5. 关键超参数与缩放直觉

| 参数 | 作用 | 常见范围 |
|------|------|----------|
| `d_model` | 表示维度 | 512-4096+ |
| `n_heads` | 子空间并行数 | 8-32 |
| `n_layers` | 深度 | 6-48+ |
| `d_ff` | FFN 容量 | 通常 `4 * d_model` |
| `dropout` | 正则化 | 0.0-0.2 |

经验：

- 固定预算下，先保证足够深度，再扩宽度
- `d_model` 必须能被 `n_heads` 整除

## 6. 训练稳定性要点

1. 学习率策略：warmup + cosine/decay
2. 损失层面：label smoothing（分类/翻译常用）
3. 梯度层面：clip grad、防 NaN
4. 规模层面：梯度累积模拟大 batch
5. 精度层面：混合精度训练降显存提吞吐

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

你要记住：Transformer 训练成败通常先由学习率与归一化策略决定。

## 7. 推理效率优化

### 7.1 KV Cache（Decoder-only 关键）

自回归生成时缓存历史 K/V，避免重复计算：

```python
past_key_values = None
for step in range(max_new_tokens):
    logits, past_key_values = model(input_ids, past_key_values=past_key_values)
```

### 7.2 高效注意力与混合精度

- Flash Attention 类实现：降低注意力内存峰值
- FP16/BF16：在稳定前提下提升吞吐

你要记住：长文本生成的延迟优化，KV cache 是第一抓手。

## 8. 常见错误与排障优先级

1. 因果掩码缺失或方向错误 -> 训练泄露未来信息
2. `train()/eval()` 切换不正确 -> Dropout 行为异常
3. Padding mask 形状不匹配 -> 无效 token 干扰注意力
4. 学习率过高 -> loss 抖动、NaN
5. 上下文过长 -> OOM，需分块或高效注意力

## 9. 最小训练骨架

```python
for step, batch in enumerate(loader):
    logits = model(batch["input_ids"], batch.get("attention_mask"))
    loss = criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

## 下一步学习

进入 [预训练模型](../pretrained-models/README.md)，理解 BERT/GPT/T5 在预训练目标、数据与微调范式上的差异。

---

**上一章**: [注意力机制](../attention-mechanisms/README.md) | **下一章**: [预训练模型](../pretrained-models/README.md)
