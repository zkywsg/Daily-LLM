# 注意力 05：多头的内部，拆开看

**读完你会知道**：多头为什么等价于“各头分别算、各自写进输出再求和”、$W_Q W_K^\top$ 与 $W_V W_O$ 各管什么、每个头能表达多少、把一个头去掉会怎样。

**前置**：[基础 B：多头](../02-multihead/)、[注意力 03：形状](../attn-03-shapes/)。

**配套**：[互动版](demo/index.html)（头贡献分解、开关消融）· [验证脚本](code/check_equivalence.py)（NumPy，已运行）· [TikTok 图卡](cards/out/)。

## 第 1 层 · 一句话

多头注意力 = 每个头独立做一次小注意力，各头结果乘上输出矩阵 $W_O$ 对应的那一块，再相加。“拼接”只是这个求和的一种写法。

## 第 2 层 · 白话

把输出矩阵 $W_O$ 横着切成 $h$ 块，每块对应一个头。第 $i$ 个头算完得到一份小结果，用自己那一块 $W_O$ 翻译成“写给总输出的内容”。所有头写的内容加在一起就是这一层的输出。所以每个头像一个小工人：自己读、自己写，写进同一块白板上。拿掉一个工人，白板上就少了他那一份。

## 第 3 层 · 手算

结构验证（用 `code/check_equivalence.py` 的设置：$n=6,\ d=8,\ h=2,\ d_k=4$，随机矩阵）：

| 检查 | 结果 |
|---|---|
| $\max\lvert \mathrm{concat}\cdot W_O-\sum_h O_hW_O^{(h)}\rvert$ | $3.6\times10^{-15}$（浮点误差，等价） |
| 每个头的分数矩阵 $S=QK^\top/\sqrt{d_k}$ 的秩 | 4、4（不超过 $d_k=4$） |
| 去掉头 0 后输出的相对变化 | 约 0.81 |

最后一行是随机矩阵、随机输入的结果，只说明“头不是摆设”；真实模型里每个头的作用差别很大，见第 4 层的文献。

## 第 4 层 · 数学推导

**(1) 求和形式**

记 $W_O=\begin{pmatrix}W_O^{(1)}\\ \vdots\\ W_O^{(h)}\end{pmatrix}$（按行切成 $h$ 块，每块 $d_k\times d$）。则

$$
\mathrm{concat}(O_1,\dots,O_h)\,W_O=\sum_{i=1}^{h}O_i\,W_O^{(i)}
$$

这是分块矩阵乘法。再代入 $O_i=A_iXW_V^{(i)}$：

$$
\text{输出}=\sum_{i=1}^{h}A_i\;X\;\underbrace{W_V^{(i)}W_O^{(i)}}_{\text{OV 电路}}
$$

**(2) 两个“电路”**

每个头的行为由两个低秩矩阵决定（Elhage et al. 2021 的视角）：

- **QK**：$W_Q^{(i)}W_K^{(i)\top}$（$d\times d$，秩 $\le d_k$）。决定“谁看谁”，即 $A_i$。
- **OV**：$W_V^{(i)}W_O^{(i)}$（$d\times d$，秩 $\le d_k$）。决定“看到之后写什么”。

**(3) 秩的限制**

$S_i=Q_iK_i^\top/\sqrt{d_k}$，其中 $Q_i,K_i$ 是 $n\times d_k$ 矩阵，所以 $\mathrm{rank}(S_i)\le d_k$。这限制的是 softmax 之前的分数矩阵；softmax 是逐元素非线性，会改变秩，所以不能据此断言权重矩阵 $A_i$ 的秩。

**(3b) 为什么 $d_k=d/h$**

每个头的参数、计算量都按 $d_k$ 缩小，$h$ 个头合起来与单头 $d_k=d$ 的开销基本相同（投影参数同为 $4d^2$）。多头是在相同预算下，把“一种匹配规则”换成“$h$ 种较小的匹配规则”。

**(4) 头的作用与去头**

- Voita et al. 2019 分析翻译模型的**编码器**自注意力：最重要的头往往承担可解释的功能（作者博客归纳为位置头、句法头、关注罕见词的头），用门控剪枝方法在英俄 WMT 上剪掉 48 个编码器头中的 38 个，只损失约 0.15 BLEU（摘要数字，已核对）。
- Michel et al. 2019（*Are Sixteen Heads Really Better than One?*）发现，测试时可以去掉大比例的头而性能没有明显下降，有的层甚至能只留 1 个头；但在编码器-解码器注意力层，剪枝的影响明显更大（摘要内容，已核对）。
- 两者结论的程度因任务和模型而异，不能推出“头都是冗余的”；也不能推出“每个头都有可解释的功能”。

## 第 5 层 · 代码与练习

```python
# 见 code/check_equivalence.py，核心三行：
parts.append(O @ Wo[sl, :])                       # 每个头对输出的贡献
concat_out = np.concatenate(heads, -1) @ Wo       # 拼接写法
sum_out = sum(parts)                              # 求和写法，二者相等
```

**练习**

1. $d=512,\ h=8$：$W_Q^{(i)}W_K^{(i)\top}$ 的形状与最大秩是多少？
2. 为什么说“拼接 + $W_O$”和“各头贡献求和”是同一件事？
3. 如果把 $d_k$ 取成 $d$（每个头都用全维），参数量怎么变？
4. 为什么只凭“去掉某个头性能不变”不能说这个头没用？

**答案要点**

1. $512\times512$，秩 $\le 64$。
2. 分块矩阵乘法：$[O_1\ \cdots\ O_h]\,W_O=\sum_i O_iW_O^{(i)}$。
3. 每头的 $W_Q,W_K,W_V$ 各 $d\times d$，参数变为 $h$ 倍的 $3d^2$，再加 $W_O$ 的 $hd\times d$，总量约为 $h$ 倍。
4. 测试集上可能没覆盖到它负责的情形，或其他头在冗余地做同一件事；缺乏变化不等于没有功能。

## 来源

- Vaswani et al., *Attention Is All You Need*, 2017。
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021（QK / OV 电路视角）。
- Voita et al., *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*, 2019。
- Michel, Levy, Neubig, *Are Sixteen Heads Really Better than One?*, 2019。
- 两篇的主要结论已对照论文摘要核对（2025-11-24 检索）；更细的数字请看原文。推导与验证脚本为本仓库原创。
