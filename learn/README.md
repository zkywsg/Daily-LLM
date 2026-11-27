# 从 0 学机器学习、深度学习与大模型

完整骨架（6 层、45 个知识块、191 个知识点的完整大纲）见 **[ROADMAP.md](ROADMAP.md)**，数据源是 [roadmap.json](roadmap.json)，改完后运行 `node learn/build-roadmap.mjs` 重新生成路线图和清单。

每个知识点配三样东西：讲解 `README.md`、互动演示 `demo/`、社媒图卡 `cards/`。

设计思路见 [../docs/superpowers/specs/2026-11-23-visual-sharing-design.md](../docs/superpowers/specs/2026-11-23-visual-sharing-design.md)。

预览：运行 `node learn/build-preview.mjs` 生成 `learn/preview.html`，把所有知识点的讲解、图卡、互动版放在一页里。

## 数学基础：线性代数

| 篇 | 内容 | 状态 |
|---|---|---|
| L01 | [向量、内积与范数](0-math/la-01-vectors/) | 已做 |
| L02 | [矩阵与线性变换](0-math/la-02-matrix/) | 已做 |
| L03 | [特征值与特征分解](0-math/la-03-eigen/) | 已做 |
| L04 | [SVD 与低秩近似](0-math/la-04-svd/) | 已做 |

每篇的脚本都实际运行过（只需 NumPy）。耗时数字随机器变化；Eckart–Young 定理按标准结论引用，原论文全文没有读。

## Transformer 完整模型

| 篇 | 内容 | 状态 |
|---|---|---|
| T01 | [残差与 LayerNorm](3-llm/tf-01-block/) | 已做 |
| T02 | [FFN 与 SwiGLU](3-llm/tf-02-ffn/) | 已做 |
| T03 | [Encoder / Decoder 三种架构](3-llm/tf-03-architectures/) | 已做 |
| T04 | [从零训练迷你 GPT](3-llm/tf-04-mini-gpt/) | 已做（训练、对照实验、梯度检查都实际运行过） |

## 注意力深度系列

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
