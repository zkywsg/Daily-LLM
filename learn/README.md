# 从 0 学机器学习、深度学习与大模型

按五层递进。每个知识点配三样东西：讲解 `README.md`、互动演示 `demo/`、社媒图卡 `cards/`。

| 层 | 内容 | 状态 |
|---|---|---|
| 0 数学 | 线性代数、概率、微积分、优化直觉 | 待写 |
| 1 传统机器学习 | 回归、树、SVM、聚类、评估 | 待写 |
| 2 深度学习基础 | 神经网络、反向传播、优化、正则化 | 待写 |
| 3 大模型 | [01 注意力基础](3-llm/01-attention/) · [02 多头](3-llm/02-multihead/) · [03 掩码与位置](3-llm/03-mask-position/) · Transformer · 预训练 · 对齐 · 推理模型 | 进行中（见下：注意力深度系列） |
| 4 工程 | 训练、量化、推理、部署 | 待写 |

设计思路见 [../docs/superpowers/specs/2026-11-23-visual-sharing-design.md](../docs/superpowers/specs/2026-11-23-visual-sharing-design.md)。

预览：运行 `node learn/build-preview.mjs` 生成 `learn/preview.html`，把所有知识点的讲解、图卡、互动版放在一页里。

## 注意力深度系列（进行中）

每篇按五层写：一句话 → 白话 → 手算 → 数学推导 → 代码与练习。

| 篇 | 内容 | 状态 |
|---|---|---|
| 00 | [动机：为什么发明注意力](3-llm/attn-01-motivation/) | 已做 |
| — | 基础 A·B·C：[注意力基础](3-llm/01-attention/) · [多头](3-llm/02-multihead/) · [掩码与位置](3-llm/03-mask-position/) | 已做 |
| 02 | [手算一遍](3-llm/attn-02-hand-calc/) | 已做 |
| 03 | [矩阵与形状](3-llm/attn-03-shapes/) | 已做 |
| 04 | [softmax 与 √d：梯度推导](3-llm/attn-04-softmax-scale/) | 已做 |
| 05 | [多头的内部：头分解与消融](3-llm/attn-05-multihead-deep/) | 已做（NumPy 脚本已运行） |
| 06 | [位置编码：正弦、RoPE、ALiBi](3-llm/attn-06-position/) | 已做 |
| 07 | [从零实现](3-llm/attn-07-from-scratch/)（7 个测试和练习） | 已做（PyTorch 版未运行） |
| 08 | [真实模型里的注意力](3-llm/attn-08-real-model/) | 脚本和查看器已写；**未在真实模型上运行**（没装 torch） |
| 09 | [高效注意力：KV cache、GQA、FlashAttention](3-llm/attn-09-efficient/) | 已做（在线 softmax 脚本已运行） |
| 10 | [综合自测与小项目](3-llm/attn-10-selftest/)（27 题） | 已做 |

说明：文献与模型配置已对照摘要或公开资料核对（见各篇“来源”）；原论文的细节数字请自行查阅。第 8 篇脚本未在真实模型上运行。
