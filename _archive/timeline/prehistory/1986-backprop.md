# 1986 · Backpropagation 反向传播

> 让多层网络真正可训练 —— 所有现代深度学习都建立在这套链式求导算法之上。

## 为什么放在前史区

反向传播不是一个时间线节点，是后来一切深度学习的「机械化求导引擎」。
从 1986 到 2025，从 MLP 到 GPT-5，每一次参数更新都是反向传播在做事。

## 它做了什么

Rumelhart、Hinton、Williams 在 *Nature* 上系统化了「用链式法则在多层网络上反向传梯度」的算法：

1. **前向**：输入沿网络前进，计算每层激活和最终 loss
2. **反向**：从 loss 出发，按拓扑逆序，每层把梯度通过该层的局部导数回传到上一层
3. **更新**：用梯度对每个参数做一步下降

## 它解决了什么

让深度网络（>2 层）从"理论上能拟合任何函数"变成"实际上可以训练"。
没有 BP，AlexNet / ResNet / Transformer 全部不存在。

## 为什么直到 2012 才爆发

- **算力**：GPU 直到 2008 后才普及
- **数据**：ImageNet 2009 才把"百万级标注图像"做成可用基准
- **细节**：ReLU / Dropout / 合适的初始化 / Batch Norm 这些工程细节直到 2010s 才齐备

也就是说：**算法（BP）等了 24 年才被工程条件成全**。这恰好就是 2012 AlexNet 的故事。

## 继续学

- [foundations/deep-learning/backpropagation/](../../foundations/deep-learning/backpropagation/) —— 反向传播的完整推导与工程实现
- [foundations/math/calculus/chain-rule.md](../../foundations/math/calculus/chain-rule.md) —— 链式法则的数学语言
- [foundations/math/calculus/autodiff-principles.md](../../foundations/math/calculus/autodiff-principles.md) —— BP 的现代工业版：自动微分
