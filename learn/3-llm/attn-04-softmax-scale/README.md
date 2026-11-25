# 注意力 04：softmax 与 √d_k，为什么要除

**读完你会知道**：softmax 的三条性质、它的梯度怎么推、为什么点积的方差是 $d_k$、为什么除以 $\sqrt{d_k}$ 恰好让训练稳定。

**前置**：[基础 A](../01-attention/) 里的分数表；会求偏导。

**配套**：[互动版](demo/index.html)（梯度矩阵、点积分布直方图、饱和对照表）· [TikTok 图卡](cards/out/)。

## 第 1 层 · 一句话

点积的大小随维度 $d_k$ 增长（标准差 $\sqrt{d_k}$），分数太大会让 softmax “饱和”，梯度几乎为 0；除以 $\sqrt{d_k}$ 把分数的标准差拉回 1。

## 第 2 层 · 白话

softmax 像一场“放大差距”的选举：分数高的拿走大部分票。维度越高，两个向量点积的随机波动越大，差距被放大得过头，一个词拿走 99%，其他词几乎 0%。这时微调任何一个分数，输出都几乎不变，也就是梯度消失，模型学不动。除以 $\sqrt{d_k}$ 就是把波动调回固定尺度，与维度无关。

## 第 3 层 · 手算

三个分数 $s=[2,1,0]$：

| 步骤 | 结果 |
|---|---|
| $e^{s}$ | $[7.389,\ 2.718,\ 1.000]$，和 $=11.107$ |
| softmax $a$ | $[0.665,\ 0.245,\ 0.090]$ |
| 分数全部加 10 | $e^{12},e^{11},e^{10}$ 同比放大，除以新的和后结果不变 |
| 分数全部乘 5 | $s=[10,5,0]$，$a=[0.993,\ 0.007,\ 0.000]$，一家独大 |

“加常数不变”叫**平移不变性**；“乘常数会变尖”说明**尺度**才是控制开关。实现里先减最大值，就是利用平移不变性防止 $e^{s}$ 溢出。

## 第 4 层 · 数学推导

**(1) softmax 的三条性质**

$$
a_i=\frac{e^{s_i}}{\sum_k e^{s_k}}
$$

1. 平移不变：$\mathrm{softmax}(s+c)=\mathrm{softmax}(s)$。
2. 输出为正且和为 1。
3. 温度：$\mathrm{softmax}(s/T)$，$T\to 0$ 趋于 one-hot，$T\to\infty$ 趋于均匀。

**(2) 梯度（Jacobian）**

$$
\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j)\quad\Longleftrightarrow\quad J=\mathrm{diag}(a)-aa^{\top}
$$

推导：对 $i=j$，$\partial a_i/\partial s_i=a_i-a_i^2=a_i(1-a_i)$；对 $i\ne j$，$\partial a_i/\partial s_j=-a_ia_j$。

关键观察：若 $a$ 接近 one-hot（某个 $a_m\to 1$，其余 $\to 0$），则每一项 $a_i(\delta_{ij}-a_j)\to 0$，$J\to 0$。**饱和等于梯度消失。**

**(3) 点积的方差**

设 $q,k\in\mathbb{R}^{d_k}$，各分量独立，均值 0，方差 1。则

$$
q\cdot k=\sum_{i=1}^{d_k}q_ik_i,\qquad \mathbb{E}[q_ik_i]=0,\quad \mathrm{Var}(q_ik_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1
$$

各项独立，方差相加：$\mathrm{Var}(q\cdot k)=d_k$，标准差 $\sqrt{d_k}$。

**(4) 为什么除以 $\sqrt{d_k}$**

$$
\mathrm{Var}\!\left(\frac{q\cdot k}{\sqrt{d_k}}\right)=\frac{d_k}{d_k}=1
$$

分数尺度与 $d_k$ 无关，softmax 输入的分布固定，梯度不随维度变小。

**(5) 前提与局限**

上面假设分量独立、方差为 1。训练后的 q、k 不再满足，所以这只是**初始化附近**的论证，用来解释设计动机，不是对训练全过程的保证。这一步的理由，有的讲解强调“梯度稳定”（Alammar），有的强调“数值稳定”（3Blue1Brown 文字版），两者是同一件事的不同侧面。

**(6) 实测（1 个 q、8 个 k，各 2000 次平均）**

| $d_k$ | 不缩放：分数标准差 | 不缩放：最大权重 | 不缩放：Jacobian 范数 | 缩放后：标准差 | 缩放后：最大权重 | 缩放后：Jacobian 范数 |
|---|---|---|---|---|---|---|
| 4 | 2.04 | 53% | 0.34 | 1.00 | 35% | 0.36 |
| 16 | 3.98 | 75% | 0.26 | 0.99 | 36% | 0.37 |
| 64 | 8.02 | 88% | 0.15 | 1.00 | 37% | 0.37 |
| 256 | 15.98 | 94% | 0.08 | 1.00 | 36% | 0.37 |
| 1024 | 31.73 | 97% | 0.04 | 1.00 | 36% | 0.37 |

（Jacobian 范数 = $J$ 的 Frobenius 范数的平均值。随机高斯向量的统计实验。“最大权重”列取自 `_shared/engine.js` 的 `scaleExperiment`（与互动页第 4 节一致）；标准差与 Jacobian 两列由另一份独立脚本得出，随机数不同，所以个别最大权重与独立脚本相差 1 个百分点。）

## 第 5 层 · 代码与练习

```python
import numpy as np

def softmax(s):
    s = s - s.max()          # 平移不变性：防溢出
    e = np.exp(s)
    return e / e.sum()

def jacobian(a):
    return np.diag(a) - np.outer(a, a)

rng = np.random.default_rng(0)
d = 64
q = rng.standard_normal(d)
K = rng.standard_normal((8, d))
s = K @ q
print("std of scores:", s.std(), " ->scaled:", (s / np.sqrt(d)).std())
print("max weight raw / scaled:", softmax(s).max(), softmax(s / np.sqrt(d)).max())
```

**练习**

1. 证明 $\mathrm{softmax}(s+c)=\mathrm{softmax}(s)$。
2. 用 $J=\mathrm{diag}(a)-aa^\top$ 验证：$J\mathbf{1}=0$（为什么？）
3. 若 $q,k$ 各分量方差为 $\sigma^2$，缩放因子应取多少才能让分数方差为 1？
4. 温度 $T=2$ 与“除以 $\sqrt{4}$”有什么关系？

**答案要点**

1. 分子分母同乘 $e^{c}$ 约掉。
2. $\sum_j a_i(\delta_{ij}-a_j)=a_i-a_i\sum_ja_j=0$。含义：所有分数同时加常数，输出不变，所以沿 $\mathbf 1$ 方向的梯度为 0。
3. $\mathrm{Var}(q\cdot k)=d_k\sigma^4$，要除以 $\sigma^2\sqrt{d_k}$。
4. 完全等价：除以 $\sqrt{d_k}=2$ 就是温度 $T=2$ 的 softmax。

## 来源

- Vaswani et al., *Attention Is All You Need*, 2017：缩放点积注意力及其脚注中对方差的说明。
- 推导与实测为本仓库原创。讲解对照参考 Jay Alammar, *The Illustrated Transformer* 与 3Blue1Brown 的注意力讲解。
