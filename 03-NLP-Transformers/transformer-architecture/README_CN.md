[English](README.md) | [中文](README_CN.md)

# Transformer 架构

## 概述

Transformer 是 2017 年在《Attention Is All You Need》论文中引入的革命性架构，它用注意力机制替代了循环，实现了并行化并在序列任务上取得了优越的性能。

## 架构

### 高层结构

```
输入嵌入 + 位置编码
    ↓
[编码器堆栈 × N]
    ↓
[解码器堆栈 × N] ← (用于 seq2seq)
    ↓
输出投影 + Softmax
```

### 编码器-解码器结构

**编码器 (Encoder)**：处理输入序列
- 多头自注意力
- 前馈网络
- 层归一化与残差连接

**解码器 (Decoder)**：生成输出序列
- 掩码多头自注意力
- 交叉注意力（到编码器）
- 前馈网络

## 核心组件

### 1. 编码器层

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力与残差连接
        attn_output = self.self_attn(x, x, x, mask)[0]
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈与残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

### 2. 解码器层

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力
        attn1 = self.masked_self_attn(x, x, x, tgt_mask)[0]
        x = self.norm1(x + self.dropout(attn1))

        # 到编码器的交叉注意力
        attn2 = self.cross_attn(x, encoder_output, encoder_output, src_mask)[0]
        x = self.norm2(x + self.dropout(attn2))

        # 前馈网络
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))

        return x
```

## 完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # 编码器堆栈
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器堆栈
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出投影
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 嵌入并添加位置编码
        src = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # 编码
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # 解码
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        # 投影到词汇表
        output = self.output_projection(tgt)
        return output
```

## 关键设计决策

### 1. 层归一化 vs 批归一化

**LayerNorm**：在特征维度归一化
- 对序列更稳定
- 与批次大小无关

```python
# LayerNorm
mean = x.mean(-1, keepdim=True)
std = x.std(-1, keepdim=True)
output = (x - mean) / (std + eps) * gamma + beta
```

### 2. 残差连接

使梯度能够流过深层网络：
```
output = LayerNorm(x + Sublayer(x))
```

### 3. 逐位置前馈网络

使用 ReLU 的两个线性变换：
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

## 变体

### 1. 仅编码器 (BERT 风格)

**应用场景**：分类、标注、表示学习

```python
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, 12, 3072)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        # 使用 [CLS] token 或平均池化
        return self.pooler(x[:, 0])
```

### 2. 仅解码器 (GPT 风格)

**应用场景**：文本生成、语言建模

```python
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # 掩码自注意力层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, 12, 3072)  # 与编码器相同但使用因果掩码
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # 因果掩码
        mask = torch.tril(torch.ones(x.size(1), x.size(1)))

        x = self.pos_encoding(self.token_embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(x)
```

## 超参数

| 参数 | 典型值 | 描述 |
|-----------|--------------|-------------|
| d_model | 512-2048 | 模型维度 |
| num_heads | 8-16 | 注意力头数 |
| num_layers | 6-24 | 编码器/解码器层数 |
| d_ff | 2048-8192 | 前馈网络隐藏层维度 |
| dropout | 0.1 | 正则化 |

## 训练技巧

### 1. 学习率调度

**预热 + 衰减**：
```python
def lr_schedule(step, warmup_steps, d_model):
    if step < warmup_steps:
        return step / warmup_steps
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
```

### 2. 标签平滑

防止过度自信：
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 3. 梯度累积

模拟大批次：
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 效率优化

### 1. KV 缓存

用于自回归生成：
```python
# 缓存之前的键和值
past_key_values = None
for token in sequence:
    output, past_key_values = model(token, past_key_values)
```

### 2. 混合精度

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Flash Attention

内存高效的注意力实现

## 应用

| 任务 | 架构 | 示例 |
|------|-------------|---------|
| 翻译 | 编码器-解码器 | 原始 Transformer |
| 分类 | 仅编码器 | BERT |
| 生成 | 仅解码器 | GPT |
| 摘要 | 编码器-解码器 | BART |
| 问答 | 仅编码器 | RoBERTa |

## 常见陷阱

1. **掩码错误**：解码器需要因果掩码
2. **缩放**：忘记注意力中的 √d_k
3. **位置编码**：未在第一层之前添加
4. **LayerNorm 位置**：Pre-LN vs Post-LN

---

**上一章**: [注意力机制](../attention-mechanisms/README.md) | **下一章**: [预训练模型](../pretrained-models/README.md)
