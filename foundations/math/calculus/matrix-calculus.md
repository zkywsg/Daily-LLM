# 矩阵微分

> 神经网络一行 `Y = X W + b` 背后的求导规则。

## 大意

矩阵微分用「分子布局 / 分母布局」两种约定，把
`∂(矩阵函数) / ∂(矩阵参数)` 系统化为可机械操作的规则。

## 待补内容

- [ ] 标量对向量、向量对向量、矩阵对矩阵 6 种情形的速查表
- [ ] 常见恒等式：`∂(Ax)/∂x = Aᵀ`、`∂(xᵀAx)/∂x = (A+Aᵀ)x` 等
- [ ] 迹技巧（trace trick）：`d tr(AXB) = BAᵀ`
- [ ] 应用：从线性层到 LayerNorm 的梯度手推

## 引用

- [反向传播](../../deep-learning/backpropagation/)
- [归一化](../../deep-learning/normalization/)
