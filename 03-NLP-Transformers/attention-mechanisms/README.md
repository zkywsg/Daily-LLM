[English](README_EN.md) | [中文](README.md)

# 注意力机制

## 概述

注意力机制让模型在生成某个位置表示时，动态选择“当前最相关的上下文”。它是 Transformer 的核心操作，也是现代 NLP/多模态模型的基础。本章面向已有机器学习基础的读者，重点讲清注意力的计算逻辑、工程细节和复杂度约束。

## 学习目标

完成本章后，你应能回答：

1. Query/Key/Value 各自承担什么角色？
2. 自注意力、交叉注意力、多头注意力分别解决什么问题？
3. 训练不稳定或显存爆炸时，优先排查哪些点？

## 1. 直觉先行：为什么注意力有效

人阅读句子时并不会平均处理每个词，而是根据当前任务有选择地聚焦。  
注意力机制把这种“聚焦”变成可学习的权重分配。

示例：

```text
句子: "The cat sat on the mat and looked at the bird"
问题: "猫在哪里?"
注意力会更集中在: "sat on the mat"
```

你要记住：注意力本质是“按相关性做加权聚合”。

## 2. 核心公式与三要素

缩放点积注意力：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

三要素角色：

| 组件 | 作用 | 类比 |
|------|------|------|
| Query | 当前要找的信息 | 检索请求 |
| Key | 可匹配的索引 | 倒排索引项 |
| Value | 被汇聚的内容 | 真实文档内容 |

缩放项 $\sqrt{d_k}$ 的作用：避免维度高时内积分布过大导致 softmax 过于尖锐、梯度变差。

你要记住：`QK^T` 给相关性，`softmax` 给权重，`@V` 给上下文聚合结果。

## 3. 三类注意力机制

### 3.1 自注意力（Self-Attention）

同一序列内部建模任意位置关系，是 Transformer 编码器/解码器块的核心。

### 3.2 交叉注意力（Cross-Attention）

`Q` 来自目标序列，`K/V` 来自源序列。  
典型于机器翻译、图文对齐、检索增强生成（RAG）中的“查询对文档”对齐。

### 3.3 多头注意力（Multi-Head Attention）

把表示空间拆为多个子空间并行计算，再拼接投影。  
不同头可关注不同关系模式（语法、语义、位置）。

你要记住：多头不是重复计算，而是“分子空间并行建模不同关系”。

## 4. 复杂度与瓶颈

设序列长度为 `n`，隐藏维为 `d`：

- 时间复杂度：`O(n^2 * d)`
- 注意力矩阵显存：`O(n^2)`

这就是长序列任务中的主要瓶颈。工程上常用：

1. 更短上下文窗口或分块
2. 高效 kernel（如 Flash Attention 实现）
3. 稀疏/线性注意力近似（按任务权衡精度）

你要记住：长序列问题通常先是显存问题，再是计算速度问题。

## 5. PyTorch 多头注意力最小实现

```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        bsz = query.size(0)

        q = self.w_q(query).view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        out = self.w_o(context)
        return out, weights
```

## 6. 掩码与位置编码

### 6.1 掩码（Mask）

必须区分两类：

1. Padding Mask：避免模型关注填充 token
2. Causal Mask：自回归任务禁止看到未来 token

掩码错误会直接造成“信息泄露”或训练失效。

### 6.2 位置编码（Positional Encoding）

注意力本身不含位置信号，需要显式注入。经典正弦位置编码：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

你要记住：没有位置编码，模型只会“看内容”，不会“看顺序”。

## 7. 注意力可视化与解释

注意力热力图可以帮助定位模型关注模式，但不要把它当作唯一解释依据。  
实践中可用于：

1. 发现掩码错误（出现不该关注的位置）
2. 检查是否过度关注标点/特殊 token
3. 辅助分析失败样本

## 8. 从注意力到 Transformer

关键转折点：仅靠注意力和前馈层就能构建高性能序列模型，且可并行训练。  
相比 RNN，Transformer 更容易扩展到大数据和大模型训练流程。

## 9. 常见陷阱与排障优先级

1. 忘记缩放项 `sqrt(d_k)` -> softmax 过尖、训练不稳
2. 掩码形状或语义错误 -> 泄露未来信息/忽略有效 token
3. 序列过长导致 OOM -> 优先降序列长度或启用高效注意力
4. 未正确 `train()/eval()` -> Dropout 行为错误
5. 混合精度下数值异常 -> 检查 loss scaling 与 `-inf` 掩码处理

## 10. 应用场景速览

| 任务 | 典型注意力模式 |
|------|----------------|
| 机器翻译 | 编码器-解码器交叉注意力 |
| 文本摘要 | 编码器自注意力 + 解码器因果注意力 |
| 图文生成/理解 | 图像 token 与文本 token 交叉注意力 |
| 语音识别 | 声学帧与文本 token 对齐注意力 |

## 下一步学习

继续进入 [Transformer 架构](../transformer-architecture/README.md)，把注意力放进完整网络块（残差、归一化、前馈层）中理解。

---

**上一章**: [序列模型](../../02-Neural-Networks/sequence-models/README.md) | **下一章**: [Transformer 架构](../transformer-architecture/README.md)
