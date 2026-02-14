[English](README_EN.md) | [中文](README.md)

# 深度学习基础

## 概述

深度学习使用具有多层的神经网络来学习数据的层次化表示。本指南涵盖神经网络基础、激活函数和训练技术。

## 神经网络基础

### 1. 从感知机到多层网络

**单个感知机**：
```
output = activation(w · x + b)
```

**多层感知机（MLP）**：
- 输入层 → 隐藏层 → 输出层
- 非线性激活实现通用近似

### 2. 前向传播

```
Layer 1: z¹ = W¹x + b¹, a¹ = σ(z¹)
Layer 2: z² = W²a¹ + b², a² = σ(z²)
Output: ŷ = a²
```

### 3. 激活函数

| 函数 | 公式 | 范围 | 特性 |
|------|------|------|------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | (0, 1) | 平滑，梯度消失 |
| **Tanh** | $\tanh(x)$ | (-1, 1) | 零中心化 |
| **ReLU** | $\max(0, x)$ | [0, ∞) | 计算高效 |
| **Leaky ReLU** | $\max(\alpha x, x)$ | (-∞, ∞) | 缓解神经元死亡 |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | (0, 1) | 多类别输出 |

## 反向传播

### 链式法则应用

```
∂L/∂W² = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂W²
∂L/∂W¹ = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂a¹ · ∂a¹/∂z¹ · ∂z¹/∂W¹
```

### 训练循环

```python
for epoch in range(num_epochs):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新权重
    optimizer.step()
```

## 实现示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# 训练设置
model = NeuralNetwork(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

## 关键概念

### 1. 权重初始化
- **Xavier/Glorot**：$\mathcal{U}(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}})$
- **He**：$\mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$ 适用于 ReLU

### 2. 归一化
- **批归一化**：归一化激活值
  - $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$
  - 稳定训练，允许更高的学习率

### 3. 正则化
- **Dropout**：训练期间随机将神经元置零
- **L2 正则化**：权重衰减 $\lambda \sum w^2$
- **早停**：当验证损失平稳时停止

## 优化算法

| 算法 | 更新规则 | 特性 |
|------|----------|------|
| **SGD** | $\theta = \theta - \alpha \nabla J$ | 简单，可能震荡 |
| **Momentum** | $v = \beta v + \nabla J$ | 加速收敛 |
| **Adam** | 每参数自适应学习率 | 最流行，鲁棒 |
| **AdamW** | Adam + 权重衰减解耦 | 更好的正则化 |

## 实践技巧

1. **学习率调度**：随时间衰减
2. **梯度裁剪**：防止梯度爆炸
3. **批次大小**：速度与稳定性之间的权衡
4. **监控**：跟踪训练/验证损失曲线

## 常见架构

- **MLP**：通用函数逼近器
- **CNN**：空间层次结构（见 CNN 架构）
- **RNN**：序列数据（见序列模型）
- **Transformer**：基于注意力（见 Transformers）

---

**上一章**：[机器学习](../machine-learning/README.md) | **下一章**：[CNN 架构](../../02-Neural-Networks/cnn-architectures/README.md)
