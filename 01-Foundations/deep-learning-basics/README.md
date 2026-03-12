[English](README_EN.md) | [中文](README.md)

# 深度学习基础

> 本章解决的真实问题是：当传统特征和线性边界开始不足以描述复杂风险模式时，神经网络为什么能学到更强的表示，以及训练为什么有时会成功、有时会失败。

## 学习目标

完成本章后，你应能回答：

1. 为什么“多层 + 非线性”能表达比线性模型更复杂的关系？
2. 反向传播到底在计算什么，它和训练循环是什么关系？
3. 当训练不稳定、loss 不下降时，优先该查哪些环节？

## 1. 为什么传统机器学习开始不够用了

延续上一章的信用风控案例。
逻辑回归可以很好地处理很多结构化任务，但当风险信号越来越依赖复杂交互时，人工特征开始变得吃力：

- 高额度本身不一定危险，但“高额度 + 利用率突增 + 近期收入波动”可能危险
- 一次逾期不一定说明问题，但“逾期发生的时机 + 消费曲线变化 + 历史还款节奏”组合起来才有意义
- 你能手工构造一些交叉特征，但很难穷尽所有组合

这就是表示学习出现的原因：模型不仅学习一个决策边界，还学习更适合任务的特征表示。

## 2. 从线性模型到神经网络

一个神经元本质上还是“线性变换 + 非线性激活”：

$$
a = \phi(w^\top x + b)
$$

如果只有线性变换，多层叠加依然可以化简成一层线性变换，表达能力不会真正提升。
关键在于加入非线性后，每一层都能把输入映射到新的表示空间。

多层感知机（MLP）的形式：

$$
h^{(1)} = \phi(W^{(1)}x+b^{(1)}), \quad
h^{(2)} = \phi(W^{(2)}h^{(1)}+b^{(2)}), \quad
\hat{y}=W^{(3)}h^{(2)}+b^{(3)}
$$

连续推演可以这样理解：

1. 第一层组合原始特征，形成局部模式
2. 第二层再组合这些局部模式，形成更高阶交互
3. 输出层基于这些表示做最终预测

你要记住：深度网络更强，不是因为“层数多”本身，而是因为它能逐层重写特征表示。

## 3. 前向传播：模型是怎么做出预测的

前向传播就是把输入按网络结构一路计算到输出：

1. 输入特征进入第一层
2. 每层先做线性变换，再过激活函数
3. 最后一层输出 logits 或预测值
4. 用损失函数衡量预测和真实标签的差距

在风控二分类任务里，常见形式是：

$$
z = W^{(L)}h^{(L-1)} + b^{(L)}, \quad
\hat{p} = \sigma(z)
$$

这里的 $\hat{p}$ 可以理解为违约概率。

你要记住：前向传播负责“给出答案”，但模型怎么学到这个答案，要看损失函数和反向传播。

## 4. 损失函数定义了“什么算学得好”

常见配对：

- 回归：线性输出 + MSE
- 二分类：Sigmoid + BCE
- 多分类：logits + CrossEntropy

二分类交叉熵的形式：

$$
L = -\big(y\log(\hat{p}) + (1-y)\log(1-\hat{p})\big)
$$

如果真实标签是违约用户，但模型给出的违约概率很低，损失就会变大。
优化器后续做的一切，本质都是在让这个损失下降。

## 5. 反向传播到底在做什么

反向传播不是额外的“黑科技”，而是链式法则在计算图上的系统应用。

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

直觉上，它做了两件事：

1. 先看最终误差有多大
2. 再把这份误差一层层分配回每个参数，问“是谁导致了这次错误”

它和代码训练循环是一一对应的：

- `loss = criterion(logits, y)`：定义误差
- `loss.backward()`：把误差反向传回去，计算所有梯度
- `optimizer.step()`：根据梯度更新参数

你要记住：反向传播不是在“找公式答案”，而是在给每个参数分配责任。

## 6. 激活函数怎么选

| 函数 | 常见用途 | 优点 | 风险 |
| --- | --- | --- | --- |
| ReLU | 隐藏层默认首选 | 简单、稳定、计算快 | 可能出现神经元死亡 |
| Leaky ReLU | ReLU 的替代 | 对负半轴保留小梯度 | 多一个超参数 |
| Sigmoid | 二分类输出层 | 可解释为概率 | 饱和时梯度变小 |
| Tanh | 早期序列模型常见 | 零中心 | 同样会饱和 |
| Softmax | 多分类输出层 | 输出归一化概率 | 需注意数值稳定性 |

隐藏层优先考虑 ReLU 系列，输出层按任务定义选。
如果你把 Sigmoid 用在很深的隐藏层，常见后果就是梯度越来越小，训练越来越慢。

## 7. 优化器与训练循环

| 优化器 | 更新直觉 | 常见场景 |
| --- | --- | --- |
| SGD | 沿当前梯度迈一步 | 强基线、大规模训练 |
| Momentum | 给梯度加惯性 | 减少震荡 |
| Adam | 为不同参数自适应学习率 | 中小模型快速起步 |
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

这段代码背后的逻辑非常重要：

- `forward` 负责产生预测
- `criterion` 负责定义偏差
- `backward` 负责计算责任归属
- `step` 负责真正调整参数

你要记住：`zero_grad -> backward -> step` 这个顺序是训练循环的骨架。

## 8. 训练稳定性：模型为什么会突然学不动

### 8.1 初始化

- Xavier/Glorot：常用于 Tanh / Sigmoid
- He 初始化：常用于 ReLU

初始化过大，激活可能爆掉；初始化过小，梯度可能传不动。

### 8.2 归一化

批归一化（BatchNorm）常见形式：

$$
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

它的价值不在于“更高级”，而在于让训练更稳、学习率更容易设。

### 8.3 正则化

- Dropout：随机屏蔽部分神经元，抑制共适应
- Weight Decay：惩罚过大的权重
- Early Stopping：验证集不再提升时及时停止

### 8.4 稳定训练的优先排查顺序

1. 数据和标签是否正确
2. 学习率是否过大或过小
3. 初始化是否合理
4. 输入尺度与归一化是否一致
5. 正则化是否过强或过弱

你要记住：训练崩掉时，先查输入和学习率，后查网络花样。

## 9. 主线案例升级：从逻辑回归到 MLP

### 9.1 为什么要升级

还是上一章的违约预测任务。
如果你怀疑“多种行为组合共同决定风险”，就可以尝试用 MLP 自动学习更复杂的特征交互。

但这不意味着深度学习一定更好。
在纯表格数据上，传统 ML 或树模型经常依然非常强，MLP 值不值得上，要看数据规模、特征类型、维护成本和收益增幅。

### 9.2 最小 PyTorch 示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RiskMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = RiskMLP(in_dim=num_features)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(num_epochs):
    logits = model(x_batch)
    loss = criterion(logits, y_batch.float())

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 9.3 怎么比较它和逻辑回归

你不应该只问“谁的 loss 更低”，而应该同时比较：

- AUC / Recall 是否真的提升
- 提升是否稳定出现在验证集和时间外样本
- 训练和部署成本是否值得
- 可解释性是否仍能满足业务和合规要求

这正是工程判断力的核心：性能不是唯一维度。

## 10. 排障与常见误区

- 现象 1：loss 一直不降
  优先查：标签、学习率、输入尺度。
- 现象 2：训练集很好，验证集变差
  优先查：过拟合、正则化、训练轮数。
- 现象 3：梯度爆炸或 NaN
  优先查：学习率、初始化、数值稳定性、梯度裁剪。
- 误区 1：网络越深越强
  正解：模型容量必须和数据规模、任务复杂度匹配。
- 误区 2：Adam 一定最好
  正解：最终还是要由验证集表现和目标指标决定。
- 误区 3：深度学习一定优于传统 ML
  正解：在结构化表格任务里，这件事经常并不成立。

## 11. 你要记住

- 深度学习的核心价值是表示学习，而不只是“参数更多”。
- 反向传播的本质，是把最终误差按链式法则分配回每个参数。
- 训练成败通常先取决于数据、学习率、初始化和归一化，而不是复杂技巧。
- 在真实工程里，深度学习是否值得使用，必须和收益、稳定性、解释性一起评估。

## 下一步学习

这一章帮你建立了最基本的神经网络直觉，但 MLP 还远不是终点。
接下来你会看到：为什么图像任务需要 CNN 的结构先验，为什么序列任务需要更适合时序依赖的建模方式。

- 进入 [CNN 架构](../../02-Neural-Networks/cnn-architectures/README.md)
- 或继续看 [序列模型](../../02-Neural-Networks/sequence-models/README.md)

---

**上一章**：[机器学习基础](../machine-learning/README.md) | **下一章**：[CNN 架构](../../02-Neural-Networks/cnn-architectures/README.md)
