# 链式法则与计算图

> 深度学习里所有反向传播的源头。

## 大意

如果 `y = f(g(x))`，那么 `dy/dx = f'(g(x)) · g'(x)`。
把任意复杂函数表达成「原子操作的计算图」之后，从输出节点反向遍历，
每一步只需要知道局部导数，乘起来就得到全局梯度。

## 待补内容

- [ ] 标量、向量、矩阵三种情形下链式法则的统一写法
- [ ] 计算图前向/反向遍历的直观例子（含手写图）
- [ ] 共享参数（如 weight tying）下的链式法则修正
- [ ] 与「自动微分」的关系：参见 [autodiff-principles.md](./autodiff-principles.md)

## 引用

- [反向传播](../../deep-learning/backpropagation/)
- 时间线节点：[1986 Backpropagation](../../../timeline/prehistory/1986-backprop.md)
