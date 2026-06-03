# 微积分：深度学习需要的最小子集

> 不写完整的微积分教材，只挑深度学习反复用到的那部分。

## 目录

- [chain-rule.md](./chain-rule.md) — 链式法则与计算图
- [gradients-and-jacobian.md](./gradients-and-jacobian.md) — 梯度、Jacobian、Hessian 的几何与代数
- [matrix-calculus.md](./matrix-calculus.md) — 矩阵 / 向量对矩阵 / 向量求导
- [autodiff-principles.md](./autodiff-principles.md) — 前向 / 反向自动微分与 PyTorch autograd

## 为什么单独成组

反向传播、优化、归一化、Transformer 注意力梯度推导都依赖这套语言。
之前散落在 `backpropagation/` 和 `linear-algebra/` 之间，引用时各章节定义不一致，
所以独立出来作为 `foundations/math/` 的第三块基石（与线代、概率信息论并列）。

## 与其他基础的关系

- 上接 [linear-algebra/](../linear-algebra/) — 矩阵微分需要的线性代数前置
- 下接 [../../deep-learning/backpropagation/](../../deep-learning/backpropagation/) — BP 是这套数学的直接应用
- 时间线引用：见 [timeline/prehistory/1986-backprop.md](../../../timeline/prehistory/1986-backprop.md)
