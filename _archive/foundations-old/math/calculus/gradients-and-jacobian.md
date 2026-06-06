# 梯度、Jacobian、Hessian

> 把"导数"从标量推广到向量和矩阵的三种工具。

## 大意

- **梯度（gradient）**：标量函数 `f: ℝⁿ → ℝ` 对每个输入变量的偏导数组成的向量
- **Jacobian**：向量函数 `f: ℝⁿ → ℝᵐ` 的一阶导数矩阵 `J ∈ ℝᵐˣⁿ`
- **Hessian**：标量函数的二阶导数矩阵，决定曲率与凸性

## 待补内容

- [ ] 梯度方向 = 最速上升方向的几何解释
- [ ] Jacobian 在反向传播中的角色（向量-Jacobian 乘积 / VJP）
- [ ] Hessian 与牛顿法 / Newton-CG / Adam 二阶近似
- [ ] 自然梯度与 Fisher 信息（衔接概率论）

## 引用

- [优化与调度](../../deep-learning/optimization-scheduling/)
- [线性代数](../linear-algebra/)
