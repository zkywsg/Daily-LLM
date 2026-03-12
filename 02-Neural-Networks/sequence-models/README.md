[English](README_EN.md) | [中文](README.md)

# 序列模型

## 概述

序列模型用于处理文本、语音、日志、时间序列等“有顺序关系”的数据。与 CNN 的局部空间归纳不同，序列模型的核心是建模“当前位置与历史上下文”的依赖。本章面向已有机器学习基础的读者，重点解释 RNN 家族为什么出现、解决了什么、又为什么被注意力机制逐步替代。

## 学习目标

完成本章后，你应能回答：

1. RNN 的状态传递机制是什么，训练难点是什么？
2. LSTM/GRU 如何通过门控缓解长期依赖问题？
3. 实战里何时继续用 RNN，何时切换到 Transformer？

## 1. RNN 的基本机制与瓶颈

基本 RNN 在每个时间步更新隐藏状态：

$$
h_t=\tanh(W_hh_{t-1}+W_xx_t+b)
$$

展开后可以看成同一组参数在时间维上重复使用，因此天然适配变长输入。  
但它有两个经典问题：

1. 梯度消失：跨长时间步反传时，梯度逐步衰减，难以学习远距离依赖。
2. 梯度爆炸：梯度累乘过大，训练不稳定。

你要记住：RNN 的能力瓶颈不是表达力不够，而是“长链路优化困难”。

## 2. LSTM：通过显式记忆通道保留长期信息

LSTM 引入细胞状态 $C_t$ 与门控机制：

- 遗忘门：保留多少旧记忆
- 输入门：写入多少新信息
- 输出门：暴露多少记忆到隐藏状态

核心更新：

$$
C_t=f_t\odot C_{t-1}+i_t\odot \tilde{C}_t,\quad
h_t=o_t\odot\tanh(C_t)
$$

设计直觉：给网络一条“相对线性”的记忆路径，让梯度不必每步都经过强非线性变换。

你要记住：LSTM 的关键收益是优化稳定性，而不只是多几个门。

## 3. GRU：更轻量的门控方案

GRU 用更新门和重置门合并了 LSTM 的部分功能：

$$
z_t=\sigma(W_z[h_{t-1},x_t]),\quad
r_t=\sigma(W_r[h_{t-1},x_t])
$$

$$
\tilde{h}_t=\tanh(W[r_t\odot h_{t-1},x_t]),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde{h}_t
$$

对比经验：

- GRU 参数更少、训练更快
- LSTM 在超长依赖任务中有时更稳
- 两者最终效果通常接近，需任务验证

你要记住：GRU 常是默认起点，LSTM 是长依赖场景的稳健备选。

## 4. LSTM vs GRU 选型表

| 维度 | LSTM | GRU |
|------|------|-----|
| 门控结构 | 3 门 + 细胞状态 | 2 门 |
| 参数量 | 较大 | 较小 |
| 训练速度 | 较慢 | 较快 |
| 长序列稳定性 | 通常更好 | 通常够用 |
| 工程默认 | 复杂场景 | 通用场景起点 |

## 5. PyTorch 最小实现

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True,
)

gru = nn.GRU(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True,
)

x = torch.randn(32, 50, 100)  # (batch, seq_len, feat_dim)
lstm_out, (h_n, c_n) = lstm(x)  # lstm_out: (32, 50, 256)
gru_out, h_n_gru = gru(x)       # gru_out:  (32, 50, 256)
```

## 6. 变长序列与文本分类实战

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens, lengths):
        x = self.embedding(tokens)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        idx = (lengths - 1).clamp(min=0)
        final = out[torch.arange(out.size(0)), idx]
        return self.fc(final)
```

要点：

- `enforce_sorted=False` 让 DataLoader 不必手动排序
- 取最后有效时间步而非直接 `out[:, -1, :]`
- 训练时配合梯度裁剪可显著提升稳定性

你要记住：RNN 项目里，变长序列处理细节往往比“换模型”更影响结果。

## 7. 为什么会过渡到注意力与 Transformer

RNN 家族的结构性限制：

1. 时序串行，难并行加速
2. 信息需跨步传递，长依赖仍有瓶颈
3. 长序列训练吞吐低

注意力机制直接建模任意位置依赖，天然并行，更适合现代大规模训练。  
详见 [注意力机制](../../03-NLP-Transformers/attention-mechanisms/README.md)。

## 8. 调参与排障优先级

1. 先检查 padding/mask/lengths 是否一致
2. 学习率过大时先降学习率再调模型
3. 使用 `clip_grad_norm_` 防梯度爆炸
4. 序列很长时优先尝试截断或分块
5. 评估时区分 token-level 与 sequence-level 指标

## 9. 何时用 RNN，何时用 Transformer

| 场景 | 推荐 |
|------|------|
| 中小数据、低延迟、模型需轻量 | GRU / LSTM |
| 超长上下文、SOTA 精度优先 | Transformer |
| 强时序先验且样本不大 | BiLSTM/BiGRU |
| 资源受限边缘设备 | 小型 GRU 或 1D-CNN |

---

**上一篇**：[CNN 架构](../cnn-architectures/README.md) | **下一篇**：[训练](../training/README.md)
