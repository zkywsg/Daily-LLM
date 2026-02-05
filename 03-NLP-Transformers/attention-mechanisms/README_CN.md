[English](README.md) | [中文](README_CN.md)

# 注意力机制

## 概述

注意力机制是使模型能够在生成输出时专注于输入相关部分的基础机制。它彻底改变了序列建模，并促成了 Transformer 架构的诞生。

## 直觉理解

### 人类注意力类比

阅读时，你不会平等地处理每个单词——你会专注于与任务相关的关键部分。

**示例**：
```
句子: "The cat sat on the mat and looked at the bird"
问题: "猫在哪里?"
注意力: 聚焦于 "sat on the mat"
```

## 核心概念

### 1. 注意力分数

衡量查询和键之间的相关性：

```
Score(Q, K) = Q · K^T / √d_k
```

### 2. 注意力权重

Softmax 归一化分数：

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

| 组件 | 作用 | 类比 |
|-----------|------|---------|
| **Query (查询)** | 你在寻找什么 | 问题 |
| **Key (键)** | 可能匹配的内容 | 索引条目 |
| **Value (值)** | 实际信息 | 内容 |

## 注意力类型

### 1. 自注意力 (Self-Attention)

每个位置关注同一序列中的所有位置。

```python
# 自注意力计算
scores = Q @ K.T / sqrt(dim)  # (seq, seq)
weights = softmax(scores)      # 注意力分布
output = weights @ V           # 加权求和
```

**应用场景**：
- 捕获句子内部关系
- 理解序列内的上下文
- Transformer 的基础

### 2. 交叉注意力 (Cross-Attention)

查询来自一个序列，键/值来自另一个序列。

**示例**：机器翻译
```
源语言 (英语): "I love machine learning"
目标语言 (法语):   "J'adore l'apprentissage automatique"
                    ↓
法语单词 "apprentissage" 关注 "learning"
```

### 3. 多头注意力 (Multi-Head Attention)

并行关注不同方面的注意力机制。

```python
# 分割为 h 个头
Q_split = split(Q, h)  # (batch, h, seq, d_k)
K_split = split(K, h)
V_split = split(V, h)

# 计算每个头的注意力
head_outputs = [attention(q, k, v) for q, k, v in zip(Q_split, K_split, V_split)]

# 拼接并投影
output = concat(head_outputs) @ W_o
```

**为什么需要多个头？**：
- 头 1：语法关系
- 头 2：语义关系
- 头 3：位置模式
- ...

## 数学表述

### 缩放点积注意力

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

其中:
- Q ∈ R^(n×d_k)  (查询)
- K ∈ R^(m×d_k)  (键)
- V ∈ R^(m×d_v)  (值)
```

**缩放因子 (√d_k)**：
防止当 d_k 较大时 softmax 进入梯度较小的区域。

### 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|-----------|------|-------|
| 注意力 | O(n²·d) | O(n²) |
| 线性层 | O(n·d²) | O(d²) |

## 实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性投影
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # 拼接头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(context), attention_weights

# 使用示例
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
output, weights = mha(x, x, x)  # 自注意力
print(f"输出形状: {output.shape}")  # (2, 100, 512)
print(f"注意力权重形状: {weights.shape}")  # (2, 8, 100, 100)
```

## 可视化注意力

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, tokens):
    """
    attention_weights: (seq_len, seq_len)
    tokens: 字符串 token 列表
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='YlOrRd')
    plt.title('注意力热力图')
    plt.show()

# 示例：显示哪些单词相互关注
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
plot_attention(weights[0, 0].detach().numpy(), tokens)
```

## 位置编码

注意力机制没有位置概念——位置编码添加此信息。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## 从注意力到 Transformer

**关键洞察**：仅凭注意力机制就足够了——无需循环或卷积。

**优势**：
1. **并行化**：同时处理所有位置
2. **长程依赖**：任意位置之间的直接连接
3. **可解释性**：注意力权重显示模型关注的内容

## 应用

### 1. 机器翻译
源语言和目标语言之间的交叉注意力

### 2. 文本摘要
自注意力识别关键句子

### 3. 图像描述生成
图像特征和文本之间的交叉注意力

### 4. 语音识别
注意力将音频帧与文本对齐

## 最佳实践

### 1. 掩码
- **填充掩码 (Padding Mask)**：忽略填充位置
- **因果掩码 (Causal Mask)**：防止关注未来（用于自回归）

### 2. 初始化
- 权重矩阵使用 Xavier/Glorot 初始化
- 对残差连接要特别注意

### 3. 优化
- 使用梯度裁剪保持稳定性
- 学习率预热

## 常见陷阱

1. **未进行缩放**：忘记 √d_k 会导致训练不稳定
2. **掩码错误**：错误的掩码导致信息泄露
3. **内存问题**：O(n²) 复杂度对长序列有问题

## 高级主题

### 稀疏注意力
- **局部注意力**：仅关注附近的位置
- **步进注意力**：关注每隔 k 个位置
- **Linformer**：通过低秩近似实现线性复杂度

### 高效 Transformer
- **Flash Attention**：内存高效的实现
- **多查询注意力**：跨头共享 K/V

---

**上一章**: [序列模型](../../02-Neural-Networks/sequence-models/README.md) | **下一章**: [Transformer 架构](../transformer-architecture/README.md)
