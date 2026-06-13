---
name: "RoPE"
year: 2021
family: "05-transformer"
order: 4
paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: ["Jianlin Su", "Yu Lu", "Shengfeng Pan", "Bo Wen", "Yunfeng Liu"]
key_idea: "把位置信息编码进 Q/K 的旋转里而不是加在 token embedding 上,attention 内积天然只依赖相对位置,长上下文外推显著更好"
---

## 前作进展

[原版 Transformer](01-transformer.md) 用绝对正余弦 PE 把位置加到输入 embedding 上,这一方案在训练长度内工作良好但有几个一直没解决的问题:

**1. 外推差**——训练长度是 2048 token 的话,推理时遇到 2049 位置就开始出问题。正余弦 PE 理论上有外推能力(因为 sin/cos 是周期函数),但实践中绝对位置加在输入上后,**模型的 attention 行为在训练长度外快速退化**。learned PE 更糟,完全不能外推。

**2. 不是真正的相对位置**——`x_i + PE_i` 和 `x_j + PE_j` 做 attention 时,模型理论上能学到 `i - j` 的依赖,但需要模型自己从大量数据中"反推"出来,效率低。

**3. PE 和 token embedding 加在一起会污染语义**——两个不同位置的同一个 token,它们的 embedding 完全相同只是 PE 不同;加起来后 attention score 既包含语义相似度也包含位置相似度,两个信号纠缠在一起。

[Transformer-XL](02-transformer-xl.md) 的相对位置编码是第一次系统性的改进——把位置信息从输入移到 attention score 里,且只用相对距离 `i - j`。但 Transformer-XL 的方案是把 attention 公式展开成 4 项,每项各自处理位置——形式复杂,工程实现要 left-shift trick。**T5** 的 relative position bias 简化成"给 attention score 加一个标量偏置",更容易实现但表达力略弱。

苏剑林(Jianlin Su)2021 年 4 月发表的 *RoFormer*(RoPE = Rotary Position Embedding)给出了一个**几何上极其优雅、工程上极其简单**的方案——**把位置信息编码进 Q/K 向量的旋转角度里**。事后看,RoPE 在表达力 + 外推性 + 实现复杂度三个维度上都达到帕累托最优,后来被 LLaMA、PaLM、Mistral、Qwen 等几乎所有现代 LLM 采用,成为 2023+ 时代的事实标准。

## 核心思想:把位置变成旋转

RoPE 的核心想法:**让 attention 的内积 `q_m^T k_n` 天然只依赖相对位置 `m - n`**,而不需要模型从绝对位置反推。

数学上要找一个函数 `f(x, m)`(把 token `x` 和位置 `m` 映射成"带位置的向量"),满足:

$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
$$

也就是说,带位置的 q 和 k 内积**只是相对距离 `m - n` 的函数**。

苏剑林给出的解是:**`f(x, m) = R_m \cdot x`,其中 `R_m` 是一个旋转矩阵**——把 x 旋转 `m·θ` 角度。具体地,在 2D 平面上:

$$
R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

为什么这能让内积只依赖相对距离?**因为旋转矩阵满足 `R_m^T R_n = R_{n-m}`**:

$$
(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
$$

这就完成了——**`q_m` 和 `k_n` 的内积自然变成了 `q` 和 `R_{n-m} k` 的内积,只依赖相对距离**。模型不需要从数据中学这件事,它是数学结构保证的。

实际中 `d_head` 是几十到上百维,RoPE 把它拆成 `d/2` 个 2D 平面,每个平面用一个不同频率 `θ_i`:

$$
\theta_i = 10000^{-2(i-1)/d}, \quad i = 1, 2, \ldots, d/2
$$

(频率公式和原版正余弦 PE 一模一样,所以"长程衰减"等性质继承下来。)

每个 2D 平面单独旋转,合起来就是给整个 `q, k` 向量一个 `d`-维旋转。代码上不需要构造完整的 `d × d` 旋转矩阵——可以利用旋转的稀疏结构在 `O(d)` 时间完成:

```python
def apply_rope(x, cos, sin):
    """对 x ∈ R^{..., d} 应用 RoPE
    cos, sin ∈ R^{seq_len, d}: 预先算好的 cos(mθ) 和 sin(mθ)
    """
    # 把 x 的最后一维拆成偶数项 x1 和奇数项 x2(两两配对成 2D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # 在每个 2D 平面上旋转
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return rotated.flatten(-2)  # 重新拼回 d 维
```

## RoPE 的三个性质

**1. 真正只依赖相对距离**——和 Transformer-XL/T5 的"在 score 上加偏置"不同,RoPE 把位置编码在 Q/K 本身里;`q^T R_{n-m} k` 是干净的相对距离形式,不需要 4 项展开或分桶查表。

**2. 长程衰减**——多频率 `θ_i = 10000^{-2(i-1)/d}` 的设计让相对位置较远时,`R_{n-m}` 在不同维度上"散开"成不相关的旋转,内积期望值下降。这模仿了人类注意力"远的词关系弱"的直觉,且不需要显式 attention mask。

**3. 外推可以做但需要调整**——原始 RoPE 在训练长度外性能会掉,但**比绝对 PE 好得多**。2023 年的 **Position Interpolation**(Chen et al.)和 **NTK-aware RoPE**(NeoX 团队)通过把 `θ_i` 缩放可以让训练在 2K 的 LLaMA 在 16K 上下文上正常工作。Meta 的 **YaRN**(2023)进一步把外推推到 128K。这些"长上下文扩展"方案都是建立在 RoPE 数学结构上的。

## Pre-LN 和 RMSNorm:配套的现代化

RoPE 在 2021–2023 流行起来的同时,Transformer block 内部的另外几件事也在同步现代化,几乎和 RoPE 一起成为现代 LLM 的标配:

**Pre-LN**(2019 Xiong, On Layer Normalization)——把 [原版 Transformer](01-transformer.md) 的 `LayerNorm(x + Sublayer(x))` 改成 `x + Sublayer(LayerNorm(x))`。LayerNorm 移到 sublayer 输入,残差直通。这让深层 Transformer 训练稳定得多,可以跳过 lr warmup,深度推到 100+ 层不掉。GPT-2 之后所有大模型都用 Pre-LN。

**RMSNorm**(2019 Zhang & Sennrich)——把 LayerNorm 的"减均值 + 除方差"简化成"只除 RMS":

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2}}
$$

去掉减均值和加偏置,参数少一半、计算快 ~7%,效果基本持平甚至略好(因为均值是冗余的)。LLaMA、Mistral 都用 Pre-RMSNorm。

**SwiGLU**(2020 Shazeer)——把 FFN 的 ReLU 换成 Gated Linear Unit + SiLU(Swish):

$$
\text{FFN}_{\text{vanilla}}(x) = \text{ReLU}(xW_1) W_2
$$

$$
\text{FFN}_{\text{SwiGLU}}(x) = (\text{SiLU}(xW_1) \odot xV) W_2
$$

多了一个 `V` 投影,FFN 参数从 2 个矩阵变 3 个,所以 `d_ff` 通常调小(`8/3 × d_model` 而不是 `4 × d_model`)保持参数量不变。SwiGLU 在所有 scaling law 实验里都比 ReLU/GeLU 略好,LLaMA 用它,GPT-4 大概率也用。

**所以一个 LLaMA 2 block 长这样**:

```python
class LlamaBlock(nn.Module):
    def forward(self, x, freqs_cos, freqs_sin):
        # Pre-RMSNorm + Attention + RoPE
        h = x + self.attn(self.norm1(x), freqs_cos, freqs_sin)
        # Pre-RMSNorm + SwiGLU FFN
        h = h + self.ffn(self.norm2(h))
        return h
```

对比原版 Transformer block——RoPE 替代正余弦 PE、Pre-RMSNorm 替代 Post-LayerNorm、SwiGLU 替代 ReLU FFN。三件事在概念上独立,但工程上一起成为 2023+ 的默认配方。

## 关键代码

完整的 RoPE attention 实现(简化版,基于 LLaMA 风格):

```python
import torch
import torch.nn as nn

def precompute_freqs(dim, max_len, base=10000.0):
    """预计算所有位置和频率的 cos/sin 表"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [d/2]
    t = torch.arange(max_len).float()                                # [seq_len]
    freqs = torch.outer(t, freqs)                                    # [seq_len, d/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1)                   # [seq_len, d]
    sin = freqs.sin().repeat_interleave(2, dim=-1)                   # [seq_len, d]
    return cos, sin

def apply_rope(x, cos, sin):
    """对 [B, h, N, d] 应用 RoPE"""
    # x_rot: 把每对相邻维度做 90 度旋转 (x_{2i+1}, -x_{2i})
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    # 注意 cos 和 sin 已经 repeat 成 d 维,直接相乘即可
    return x * cos + x_rot * sin

class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=4096):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        cos, sin = precompute_freqs(self.d_k, max_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = [t.view(B, N, self.h, self.d_k).transpose(1, 2)
                   for t in qkv.chunk(3, dim=-1)]
        # 关键:对 Q 和 K 应用 RoPE(V 不动)
        cos, sin = self.cos[:N], self.sin[:N]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # 后面就是标准 scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)
```

注意几个工程要点:

- **只对 Q 和 K 应用 RoPE,V 不动**——V 是承载内容信息的,不应该被位置污染
- **cos/sin 可以预计算并缓存**——它们只依赖位置和频率,不依赖输入
- **`x_rot` 那一行实现的是 90 度旋转的等价形式**——`(x_{2i}, x_{2i+1}) → (-x_{2i+1}, x_{2i})`,然后 `x*cos + x_rot*sin` 就是完整的 RoPE 旋转

整个 RoPE 实现 ~10 行,没有额外参数(`cos/sin` 是 buffer 不是 parameter),推理时也没有额外计算开销——这是 RoPE 战胜其他相对位置方案的核心工程优势。

## 影响 / 后续

RoPE 在 2021 年发表后两年内被几乎所有现代 LLM 采用:

| 模型 | 位置编码 | 发布 |
|------|------|------|
| GPT-3(2020) | learned absolute | 2020 |
| BERT(2018) | learned absolute | 2018 |
| T5(2019) | relative bias(分桶) | 2019 |
| GPT-NeoX(2022) | **RoPE** | 2022 |
| LLaMA(2023) | **RoPE** | 2023 |
| LLaMA 2 / 3(2023–24) | **RoPE** | 2023 |
| PaLM(2022) | **RoPE** | 2022 |
| Mistral / Mixtral(2023) | **RoPE** | 2023 |
| Qwen(2023) | **RoPE** | 2023 |
| GPT-4 / Claude 3(推测) | 大概率 RoPE 或类似 | 2023–24 |

唯一持续使用其他方案的主流模型是 BLOOM 和 MPT,它们用 **ALiBi**(Press 2021)——不用 PE,直接给 attention score 加一个与 `|i-j|` 成正比的负偏置。ALiBi 在外推性上更激进(理论上无限长),但表达力略弱,主流选择仍是 RoPE。

RoPE 的成功也催生了 2023 年的 **长上下文扩展**子领域。原版 LLaMA 训练长度 2K,但通过 RoPE 的几个简单 trick 就能扩到长得多的上下文:

- **Position Interpolation(PI)** —— 把 RoPE 的频率缩小 `L_new / L_train` 倍,等价于"压缩位置坐标轴"。少量微调就能让 LLaMA 2K → 16K
- **NTK-aware Scaling** —— 不同频率维度用不同缩放因子,高频(短程)保持原样,低频(长程)按比例缩放
- **YaRN**(Yet another RoPE extensioN)—— 组合 PI + NTK + 温度调整,LLaMA 2K → 128K 的最强方案

这些方法都是建立在 RoPE 的旋转数学结构上的——**绝对 PE 和 learned PE 上几乎不存在等效的"上下文扩展"方法**。从这个角度,RoPE 不只是一个位置编码,它是**长上下文 LLM 的基础设施**。

→ [05-flash-attention.md](05-flash-attention.md) · 与 RoPE 完全正交,attention 算法 + RoPE 位置 = 现代 LLM 标配
→ [02-transformer-xl.md](02-transformer-xl.md) · 相对位置思想的早期推广
→ [03-sparse-attention.md](03-sparse-attention.md) · 长上下文的另一条路线,可以和 RoPE 组合
→ [01-transformer.md](01-transformer.md) · 父结构,原版绝对 PE 被本节点取代
→ [../07-gpt-scaling/](../07-gpt-scaling/) · LLaMA / GPT 系都用 RoPE
