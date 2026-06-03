# 自动微分原理

> 不是数值微分（有限差分），不是符号微分（公式展开），是「计算图 + 链式法则的机械化」。

## 大意

- **前向模式（forward-mode）**：从输入沿计算图前进，同时维护值和导数。适合输入维度小、输出维度大的场景。
- **反向模式（reverse-mode）**：先前向把所有中间值算出来，再反向遍历用链式法则把梯度回传。深度学习用的是这种 —— 因为输出（loss）是标量、输入（参数）维度很高。
- **PyTorch autograd / JAX grad** 都是反向模式的工程实现：
  - 前向时构建动态计算图
  - 调用 `.backward()` 时对图做拓扑逆序遍历
  - 每个 op 注册一个 `vjp`（向量-Jacobian 乘积）

## 待补内容

- [ ] 前向 vs 反向的计算/内存开销对比
- [ ] checkpointing（用算力换内存）
- [ ] 高阶导数与 `create_graph=True`
- [ ] JIT / XLA 编译对自动微分的影响

## 引用

- [反向传播](../../deep-learning/backpropagation/)
- [chain-rule.md](./chain-rule.md)
