[English](README_EN.md) | [中文](README.md)

# 序列模型

## 概述

序列模型用于处理文本、时间序列和语音等序列数据。本指南涵盖了 RNN、LSTM、GRU，以及向注意力机制的过渡。

## 循环神经网络（RNN）

### 1. 基本 RNN

**循环连接**：
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

**展开视图**：
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T → Output
     ↑_________↑_________↑
```

### 2. 基本 RNN 的问题

**梯度消失（Vanishing Gradients）**：
- 梯度在时间步上呈指数级缩小
- 无法捕捉长期依赖关系

**梯度爆炸（Exploding Gradients）**：
- 梯度呈指数级增长
- 导致训练不稳定

## 长短期记忆网络（LSTM）

### 单元结构

**门控信息流**：

1. **遗忘门（Forget Gate）**：决定丢弃什么
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```

2. **输入门（Input Gate）**：决定存储什么
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```

3. **更新单元状态**：
   ```
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
   ```

4. **输出门（Output Gate）**：决定输出什么
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(C_t)
   ```

### LSTM 变体

| 变体 | 修改 | 优势 |
|---------|-------------|---------|
| **Peephole** | 门控可见单元状态 | 更好的时序控制 |
| **Coupled** | 遗忘门和输入门合并 | 参数更少 |
| **BiLSTM** | 双向处理 | 获取未来上下文 |

## 门控循环单元（GRU）

### 简化架构

**两个门控**（相比 LSTM 的三个）：

1. **更新门（Update Gate）**：平衡新旧信息
   ```
   z_t = σ(W_z · [h_{t-1}, x_t])
   ```

2. **重置门（Reset Gate）**：决定遗忘多少过去的信息
   ```
   r_t = σ(W_r · [h_{t-1}, x_t])
   ```

3. **候选激活**：
   ```
   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
   ```

4. **最终更新**：
   ```
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
   ```

### LSTM vs GRU

| 方面 | LSTM | GRU |
|--------|------|-----|
| 门控数量 | 3 | 2 |
| 参数量 | 较多 | 较少（少 25%） |
| 性能 | 相近 | 相近 |
| 速度 | 较慢 | 较快 |
| 适用场景 | 长序列 | 通用场景 |

## 实现

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(
    input_size=100,    # 嵌入维度
    hidden_size=128,   # 隐藏状态维度
    num_layers=2,      # 堆叠 LSTM
    batch_first=True,  # 输入格式：(batch, seq, feature)
    dropout=0.3,       # 层间 Dropout
    bidirectional=True # 双向处理
)

# 前向传播
input_seq = torch.randn(32, 50, 100)  # (batch, seq_len, features)
output, (hidden, cell) = lstm(input_seq)
# output: (32, 50, 256) - 拼接的前向+后向输出

# GRU（更简单，参数更少）
gru = nn.GRU(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

output, hidden = gru(input_seq)
```

## 应用

### 1. 文本分类
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 使用最后的隐藏状态
        final_hidden = lstm_out[:, -1, :]
        return self.fc(final_hidden)
```

### 2. 序列到序列（Sequence to Sequence）
用于机器翻译的编码器-解码器架构

### 3. 时间序列预测
金融预测、天气预报

## 局限性与过渡

### RNN 的不足之处

1. **串行处理**：无法并行化
2. **长期依赖**：信息瓶颈
3. **训练缓慢**：逐步计算

### 解决方案：注意力机制与 Transformer

**注意力机制**：
- 任意位置之间的直接连接
- 无串行依赖
- 并行计算

详见 [注意力机制](../../03-NLP-Transformers/attention-mechanisms/README.md) 了解 RNN 之后的演进。

## 最佳实践

### 1. 处理变长序列
```python
# 填充和打包
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 按长度排序
lengths = [len(seq) for seq in sequences]
packed = pack_padded_sequence(padded_seqs, lengths, batch_first=True)
output, _ = lstm(packed)
output, _ = pad_packed_sequence(output, batch_first=True)
```

### 2. 正则化
- **Dropout**：层间使用（不在循环连接上）
- **权重衰减**：L2 正则化
- **梯度裁剪**：防止梯度爆炸

### 3. 初始化
- 权重使用 Xavier/Glorot 初始化
- 循环权重使用正交初始化

## 如何选择

| 场景 | 推荐 |
|----------|---------------|
| **短序列（<50）** | GRU（更快） |
| **长序列（>100）** | LSTM（更好的记忆能力） |
| **需要双向处理** | BiLSTM/BiGRU |
| **追求最高精度** | Transformer（见下一章） |
| **资源受限** | GRU 或 CNN |

---

**上一篇**：[CNN 架构](../cnn-architectures/README.md) | **下一篇**：[训练](../training/README.md)
