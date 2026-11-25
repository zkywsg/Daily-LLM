# 注意力 06：位置编码，正弦、RoPE、ALiBi

**读完你会知道**：为什么注意力需要位置信息、四类主流方案各自怎么做、RoPE 为什么“只看相对距离”、ALiBi 的斜率怎么来。

**前置**：[基础 C：掩码与位置](../03-mask-position/) 里“打乱词序权重不变”的实验。

**配套**：[互动版](demo/index.html)（RoPE 旋转、距离曲线、ALiBi 偏置）· [TikTok 图卡](cards/out/)。

## 第 1 层 · 一句话

注意力对词序“看不见”，位置要靠额外手段注入：把位置向量加到输入上（正弦 / 可学习），或者直接改 q、k（RoPE 旋转），或者给分数加一个随距离变化的偏置（ALiBi）。

## 第 2 层 · 白话

- **加法派**：给每个位置一个编号向量，加到词向量上。简单，但“位置 5 和位置 8 差 3”这件事要靠模型自己从两个编号里悟出来。
- **旋转派（RoPE）**：不加编号，而是把 q、k 按位置旋转一个角度。两个向量旋转后再点积，结果只取决于角度差，也就是只取决于相对距离。
- **偏置派（ALiBi）**：不碰向量，直接在分数表上减一个“距离 × 斜率”，离得越远扣得越多。

## 第 3 层 · 手算

**RoPE，二维一对分量**。取 $q=[1,\,0.3]$，$k=[0.6,\,0.9]$，$\theta=0.5$ 弧度。位置 $m$ 的向量旋转角度 $m\theta$：

| $(m,n)$ | $m-n$ | $\langle R(m\theta)q,\ R(n\theta)k\rangle$ |
|---|---|---|
| (5, 2) | 3 | 0.7797 |
| (10, 7) | 3 | 0.7797 |
| (103, 100) | 3 | 0.7797 |
| (3, 0) | 3 | 0.7797 |

距离相同，点积相同，与绝对位置无关。

**ALiBi，8 个头的斜率**：$m_k=2^{-8k/8}$，$k=1..8$，即 $\tfrac12,\tfrac14,\tfrac18,\dots,\tfrac1{256}$。第 $i$ 个词看第 $j$ 个词（$j\le i$）时，分数减去 $m_k(i-j)$。斜率大的头只看近处，斜率小的头能看远。

## 第 4 层 · 数学推导

**(1) 正弦绝对位置编码**（Vaswani 2017）

$$
PE(pos,2j)=\sin\!\big(pos/10000^{2j/d}\big),\qquad PE(pos,2j+1)=\cos\!\big(pos/10000^{2j/d}\big)
$$

加到词向量上：$x_{pos}\leftarrow x_{pos}+PE(pos)$。不同维度对应不同频率，像一把多刻度的尺。

**(2) RoPE**（Su et al. 2021）

把 $d$ 维向量拆成 $d/2$ 对，每对看作二维平面上的点，第 $i$ 对用角频率 $\theta_i=10000^{-2i/d}$（$i=0,\dots,d/2-1$）。位置 $m$ 处把每一对旋转 $m\theta_i$：

$$
R(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\ \sin\alpha&\cos\alpha\end{pmatrix},\qquad \tilde q_m=R(m\theta)q,\quad \tilde k_n=R(n\theta)k
$$

关键性质（旋转矩阵满足 $R(a)^{\top}R(b)=R(b-a)$）：

$$
\tilde q_m^{\top}\tilde k_n=q^{\top}R(m\theta)^{\top}R(n\theta)\,k=q^{\top}R\big((n-m)\theta\big)\,k
$$

结果只依赖 $n-m$。完整实现里，每一对用各自的 $\theta_i$ 分别做，总分数是各对之和。注意旋转只作用在 q、k 上，不作用在 v 上。

**(3) 多频率与距离**

高频对（$\theta_i$ 大）转得快，分辨近距离；低频对（$\theta_i$ 小）转得慢，分辨远距离。互动页第 3 节画的是 $q=k=\mathbf 1$ 时，分数随距离 $\Delta$ 的变化 $\sum_i\cos(\Delta\theta_i)$，它在距离增大时整体衰减、有振荡。这是特殊设定下的演示，不代表真实 q、k 的行为。

**(4) ALiBi**（Press et al. 2021）

$$
\mathrm{score}_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}-m_h\,(i-j),\qquad j\le i
$$

斜率几何级数：$h$ 个头时 $m_k=2^{-8k/h}$（$k=1..h$）。论文给出的是头数为 2 的幂时的公式，非 2 的幂时各实现的处理不同（有的在相邻两个 2 的幂之间插值），所以互动页拖到 3、5、6 等头数时只是套用同一个闭式，不代表某个具体实现。无可学习参数，论文的主要结论是训练在较短序列上、推理时外推到更长序列的表现更好。

**(5) 方案对比**

| 方案 | 怎么注入 | 参数 | 典型使用（需自行核对具体版本） |
|---|---|---|---|
| 正弦绝对 | 加到输入 | 无 | 原始 Transformer |
| 可学习绝对 | 加到输入，查表 | $L_{max}\times d$ | BERT、GPT-2 一类 |
| RoPE | 旋转 q、k | 无 | LLaMA 系列等 |
| ALiBi | 分数加偏置 | 无 | BLOOM、MPT 等 |

## 第 5 层 · 代码与练习

```python
import numpy as np

def rope(x, pos, base=10000.0):
    # x: [d]，d 为偶数；pos: 整数位置
    d = x.shape[-1]
    theta = base ** (-2 * np.arange(d // 2) / d)
    a = pos * theta
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(a) - x2 * np.sin(a)
    out[1::2] = x1 * np.sin(a) + x2 * np.cos(a)
    return out

rng = np.random.default_rng(0)
q, k = rng.standard_normal(8), rng.standard_normal(8)
print(rope(q, 5) @ rope(k, 2), rope(q, 10) @ rope(k, 7))   # 距离同为 3，应相等

def alibi_bias(n, h):
    slopes = 2.0 ** (-8 * np.arange(1, h + 1) / h)
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    return -slopes[:, None, None] * np.maximum(i - j, 0)      # [h, n, n]
```

**练习**

1. 证明 $R(a)^\top R(b)=R(b-a)$。
2. $d=8$ 时，RoPE 的四个 $\theta_i$ 分别是多少？
3. RoPE 为什么不对 v 做旋转？
4. 8 个头的 ALiBi，哪个头最“近视”？
5. 把 ALiBi 偏置加在因果掩码之前还是之后？

**答案要点**

1. 展开矩阵乘法，用 $\cos a\cos b+\sin a\sin b=\cos(b-a)$ 等和差公式。
2. $1,\ 0.1,\ 0.01,\ 0.001$。
3. 位置信息只需要影响“谁和谁匹配”，也就是 q·k；v 是被取走的内容，旋转它会让输出带上位置扰动。
4. 斜率最大（1/2）的头，每远一格扣 0.5 分，几乎只看邻近。
5. 先加偏置再做掩码（掩码是把未来设为 −∞，加在最后，保证未来仍是 0 权重）；这两个操作可交换，只要掩码最终生效即可。

## 来源

- Vaswani et al., *Attention Is All You Need*, 2017：正弦位置编码。
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021：RoPE。
- Press, Smith, Lewis, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*, 2021：ALiBi。
- RoPE 的 $\theta_i=10000^{-2i/d}$ 取自论文及后续使用它的工作（已核对）；ALiBi 的斜率公式已核对。表中“典型使用”来自这些方法的公开介绍，具体到某个模型的版本请查其论文或模型卡。推导、手算与代码为本仓库原创；NumPy 代码未运行单元测试，数值由 JS 独立核对（见手算表）。
