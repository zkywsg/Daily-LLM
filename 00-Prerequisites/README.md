# Phase 00 · 前置准备

进入视觉线和语言线之前所需的最低限度神经网络基础。
不包含经典机器学习算法（SVM、决策树、K-Means 等）。

## 本阶段内容

### [深度学习基础](deep-learning-basics/README.md)
- 神经元与前向传播
- 反向传播与梯度下降
- 激活函数：Sigmoid、ReLU 及其变体
- 损失函数与训练循环

### [反向传播与优化器](backpropagation/README.md)
- 链式法则与计算图推导
- 梯度消失/爆炸与对策
- 优化器对比：SGD、Momentum、Adam
- 训练循环骨架与学习率调度

### [激活函数家族](activation-functions/README.md)
- Sigmoid 的梯度消失与 ReLU 的崛起
- Dying ReLU 问题与变体（Leaky ReLU、GELU 等）
- 激活函数选择策略

### [正则化与 Dropout](regularization/README.md)
- 过拟合的诊断与根源
- Dropout 核心机制与变体（DropConnect、SpatialDropout、DropBlock）
- 权重衰减（L2 正则化）
- 其他正则化手段：Early Stopping、Data Augmentation、Label Smoothing

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2012 | ReLU 普及 | 取代 sigmoid，梯度流动恢复，训练速度提升数倍 |
| 2012 | GPU 深度学习生态 | CUDA 加速训练，计算基础设施确立 |
| 2014 | Adam 优化器 | 几乎不需要调学习率的默认优化器 |
| 2015 | Batch Normalization | 训练速度提升数量级，允许更高学习率 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [视觉线 →](../01-Visual-Intelligence/)
