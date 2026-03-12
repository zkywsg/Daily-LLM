[English](README_EN.md) | [中文](README.md)

# 深度学习基础

## 概述

深度学习可以看成“可学习的函数组合”：每一层把输入变换成更抽象的表示，最终完成分类、回归或生成任务。本章面向已有机器学习基础的读者，目标是把你从“会用模型”推进到“理解训练为何有效、何时失效”。

## 学习目标

完成本章后，你应能回答：

1. 为什么多层+非线性能表达复杂函数？
2. 反向传播到底在计算什么？
3. 训练不收敛时，优先检查哪些环节？

## 1. 从线性模型到多层网络

单个神经元本质是线性变换加非线性：

$$
a = \phi(w^\top x + b)
$$

如果没有非线性激活，多层线性层可合并为一层线性层，表达能力不会提升。深度学习的关键不只是“更深”，而是“线性变换 + 非线性”交替堆叠。

**多层感知机（MLP）**可写成：

$$
h^{(1)} = \phi(W^{(1)}x+b^{(1)}), \quad
h^{(2)} = \phi(W^{(2)}h^{(1)}+b^{(2)}), \quad
\hat{y}=W^{(3)}h^{(2)}+b^{(3)}
$$

你要记住：深度网络通过逐层特征重编码，把“难问题”转成更容易线性分割的问题。

## 2. 前向传播与损失函数

前向传播就是“按图计算”得到预测 $\hat{y}$，再用损失函数衡量偏差。

常见配对：

- 回归：线性输出 + MSE
- 二分类：Sigmoid + BCE
- 多分类：logits + CrossEntropy（内部含 Softmax）

你要记住：损失函数定义了“什么是好模型”，优化器只是在最小化这个目标。

## 3. 反向传播在做什么

反向传播不是神秘算法，而是链式法则在计算图上的系统化应用。

以两层网络为例：

$$
\frac{\partial L}{\partial W^{(2)}}=
\frac{\partial L}{\partial z^{(2)}}\frac{\partial z^{(2)}}{\partial W^{(2)}},
\quad
\frac{\partial L}{\partial W^{(1)}}=
\frac{\partial L}{\partial z^{(2)}}
\frac{\partial z^{(2)}}{\partial h^{(1)}}
\frac{\partial h^{(1)}}{\partial z^{(1)}}
\frac{\partial z^{(1)}}{\partial W^{(1)}}
$$

含义是：输出误差逐层“归因”回每个参数，然后参数按梯度方向更新。

你要记住：梯度是“局部变化率”，反向传播是把局部变化率拼成全局方向。

## 4. 激活函数怎么选

| 函数 | 公式 | 常见用途 | 风险 |
|------|------|----------|------|
| ReLU | $\max(0, x)$ | 隐藏层默认首选 | 神经元死亡 |
| Leaky ReLU | $\max(\alpha x, x)$ | ReLU 不稳定时替代 | 额外超参 |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 二分类输出层 | 饱和导致梯度小 |
| Tanh | $\tanh(x)$ | 早期序列模型 | 同样可能饱和 |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | 多分类输出解释为概率 | 数值稳定性需注意 |

你要记住：隐藏层优先 ReLU 系列，输出层按任务定义选择。

## 5. 训练稳定性的三件套

### 5.1 初始化

- Xavier/Glorot：适合 Tanh/Sigmoid
- He 初始化：适合 ReLU

初始化过大或过小都会导致梯度传播困难。

### 5.2 归一化

批归一化（BatchNorm）核心公式：

$$
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

作用是缓解内部协变量偏移，通常让训练更稳、可用更大学习率。

### 5.3 正则化

- Dropout：随机失活，抑制共适应
- L2/Weight Decay：惩罚大权重
- 早停：验证集长期不提升时停止

你要记住：不稳定先看初始化与学习率，过拟合再加正则化。

## 6. 优化器与训练循环

| 优化器 | 更新直觉 | 典型场景 |
|--------|----------|----------|
| SGD | 沿当前梯度走一步 | 大规模训练基线 |
| Momentum | 梯度加“惯性” | 减少震荡 |
| Adam | 自适应一阶/二阶矩 | 中小模型快速收敛 |
| AdamW | Adam + 解耦权重衰减 | 现代默认常用 |

标准训练循环：

```python
for epoch in range(num_epochs):
    model.train()
    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

你要记住：`zero_grad -> backward -> step` 顺序不要错。

## 7. 最小可运行示例（PyTorch）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(10):
    logits = model(images)          # images: [B, 784]
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1:02d} | Loss {loss.item():.4f}")
```

## 8. 调参优先级（实战）

当结果不理想时，建议按这个顺序排查：

1. 数据与标签是否正确（先排除数据问题）
2. 学习率是否过大/过小（最常见）
3. 模型容量是否不足或过大
4. 正则化强度是否合适
5. 批大小与训练时长是否匹配

## 9. 常见误区

- 误区 1：只看训练损失下降  
  正解：同时看验证集指标，防止过拟合。
- 误区 2：盲目加深网络  
  正解：先确保数据规模与任务复杂度支持模型容量。
- 误区 3：默认 Adam 就一定最好  
  正解：最终要用验证集与目标指标比较。

## 下一步学习

- 想理解图像任务的结构先验：进入 [CNN 架构](../../02-Neural-Networks/cnn-architectures/README.md)
- 想理解序列建模：进入 [序列模型](../../02-Neural-Networks/sequence-models/README.md)
- 想理解注意力体系：进入 [Transformer 架构](../../03-NLP-Transformers/transformer-architecture/README.md)

---

**上一章**：[机器学习](../machine-learning/README.md) | **下一章**：[CNN 架构](../../02-Neural-Networks/cnn-architectures/README.md)
