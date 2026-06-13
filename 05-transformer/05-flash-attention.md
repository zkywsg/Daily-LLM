---
name: "FlashAttention"
year: 2022
family: "05-transformer"
order: 5
paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Ré"]
key_idea: "把 attention 从 HBM 搬到 SRAM 算,分块 + 重计算把 O(N²) 显存压成 O(N) 而结果完全等价,attention 训练/推理快 2-4× 且支持更长序列"
---

## 前作进展

到 2022 年初,Transformer 在 NLP / 视觉 / 多模态全面统治后,长上下文成为最持续的瓶颈。前面三个节点各自从算法层面尝试解决:

- [Transformer-XL](02-transformer-xl.md) — segment cache,但段内仍 O(L²)
- [Sparse Attention](03-sparse-attention.md) — O(N) 但需要稀疏模式设计 + CUDA kernel
- [RoPE](04-rope.md) — 改善 PE 外推,但 attention 本身复杂度没变

但所有这些方案都在**算法层面**改 attention 的数学定义。Tri Dao 等人在斯坦福(Christopher Ré 组)从一个完全不同的角度看这件事:**原版 dense attention 慢/吃显存,不是因为 FLOPs 多,而是因为 HBM 访存太多**。

这个观察来自对 GPU 内存层级的认真分析:

| 层级 | 容量 | 带宽 | 延迟 |
|------|------|------|------|
| **SRAM**(片上,L1/L2 cache 级别) | 20 MB(A100) | **19 TB/s** | ~10 ns |
| **HBM**(片外,主显存) | 40-80 GB | **1.5-2 TB/s** | ~400 ns |

SRAM 比 HBM 快 **10×**,但容量小 **3000×**。当你做 attention 时:

```python
# 朴素实现
S = Q @ K.T          # [N, N] — 要写到 HBM
P = softmax(S)       # 读 S,写回 [N, N] — 又一次 HBM 来回
O = P @ V            # 读 P 和 V,写 O — 再一次 HBM 来回
```

每个 attention 都要把 `N × N` 矩阵在 HBM 上**物化三次**,N=4096 时是 `4096² × 2 bytes × 3 次 ≈ 100 MB / head / layer / batch`。GPU 的算力(312 TFLOPs A100)远远大于带宽能喂的数据量,**计算单元大部分时间在等内存**,GPU 利用率只有 20–30%。

更糟的是,反向传播需要 `S` 和 `P` 重新算梯度,所以要把它们**存到 HBM 等反传**——`O(N²)` 显存代价。

FlashAttention 的核心观察:**`N × N` 矩阵根本不应该物化,attention 应该完全在 SRAM 里算**。

## 核心思想:分块 + online softmax + 重计算

直接把 attention 搬到 SRAM 有两个明显障碍:

**1. SRAM 装不下整个 `N × N` 矩阵**——A100 的 SRAM 是 20 MB,但 N=4096 时 `N × N × 2 bytes = 32 MB`,放不下
**2. Softmax 需要看整行**——`softmax(s_i)` 要先算 `max(s_i)` 和 `sum(exp(s_i - max))`,这些是行级 reduce,看似必须先把整行算完才能 softmax

FlashAttention 用三个 trick 一起解决:

**Trick 1:分块**——把 Q、K、V 都按行切成 block(典型 block size 128–256),每次只把一对 Q-block 和 K/V-block 加载到 SRAM 里:

```
对每个 Q_i (Q 的第 i 个 block):
    对每个 K_j, V_j (K/V 的第 j 个 block):
        把 Q_i, K_j, V_j 加载到 SRAM
        在 SRAM 里算 S_ij = Q_i @ K_j.T
        更新输出 O_i 和 softmax 统计量
```

**Trick 2:Online softmax**——这是技术核心。普通 softmax 要看整行,但有一个数学性质允许"流式更新":

如果我已经算了前 `j` 个 block 的部分输出 `O_i^{(j)}` 和归一化因子 `(m_i^{(j)}, \ell_i^{(j)})`(分别是 running max 和 running sum-of-exp),来了一个新 block,可以**精确更新**而不需要重算前面:

$$
m_i^{(\text{new})} = \max(m_i^{(j)}, \max(S_{i, j+1}))
$$

$$
\ell_i^{(\text{new})} = e^{m_i^{(j)} - m_i^{(\text{new})}} \ell_i^{(j)} + \sum_k e^{S_{i, j+1, k} - m_i^{(\text{new})}}
$$

$$
O_i^{(\text{new})} = \frac{\ell_i^{(j)} e^{m_i^{(j)} - m_i^{(\text{new})}}}{\ell_i^{(\text{new})}} O_i^{(j)} + \frac{1}{\ell_i^{(\text{new})}} \sum_k e^{S_{i, j+1, k} - m_i^{(\text{new})}} V_{j+1, k}
$$

(`m` 用减最大值的数值稳定 softmax;`\ell` 是分母 sum-of-exp;每收到新 block 用一次 rescale 把旧 partial 更新成与新 max 一致。)

最终结果**完全等价于一次性 softmax**——这不是近似,FlashAttention 是 **exact attention**。

**Trick 3:反传时重计算 S 和 P**——反传需要 `S` 和 `P`,正常做法是 forward 时存 HBM。FlashAttention 选择**反传时重算**——只在 HBM 存最终输出 `O`、softmax 统计量 `(m, \ell)`、Q/K/V(原本就有)。反传时用这些重新走一遍 forward 的分块计算,把 `S` 和 `P` 算回来。

代价:反传 FLOPs 多 2.5× ——但因为大部分时间被内存带宽卡住,**实际 wall-clock 时间反而短了**。这是经典的"以计算换内存"trade-off。

## 性能数据

FlashAttention 的论文报告(A100,fp16):

| 序列长度 | 朴素 PyTorch attention | FlashAttention | 加速 |
|------|------|------|------|
| 512 | 1.4 ms | 0.4 ms | 3.5× |
| 1024 | 5.7 ms | 1.0 ms | 5.7× |
| 2048 | 22.8 ms | 2.4 ms | 9.5× |
| 4096 | OOM | 7.1 ms | ∞ |
| 8192 | OOM | 26.8 ms | ∞ |

**显存:O(N²) → O(N)**。N=8K 时朴素实现要 256 MB / head / layer / batch,FlashAttention 只要 64 KB / head / layer / batch——少 **4000×**。

端到端训练:GPT-2 medium 训练快 **2.4×**(因为模型里其他部分也吃时间),BERT-large 训练快 **15%**。看起来不夸张,但这个加速是**免费的、完全等价**——和 sparse attention 用精度换速度不同。

## 三个版本的演化

- **FlashAttention v1**(2022 May)—— Tri Dao 单作者的原版,确立分块 + online softmax + 重计算的核心算法
- **FlashAttention v2**(2023 July)—— 重排循环顺序、更好的 warp 分工,A100 上再快 **2×**(理论极限的 ~70%)
- **FlashAttention v3**(2024 July)—— H100-specific 优化,利用 Hopper 的 WGMMA 和异步执行,H100 上达到理论极限的 **75%**;fp8 路径下达到 **1.2 PFLOPs/s**

每一代都是"算法不变 + 硬件特定优化"——这种持续投入也反映了 FlashAttention 在 LLM 工业链里的核心地位:**几乎所有现代 LLM 训练/推理都跑在 FlashAttention 上**。

## GQA / MQA:推理时代的多头演化

FlashAttention 让训练时的 attention 不再是瓶颈,但**推理时**(尤其长上下文生成)又冒出新瓶颈——**KV cache**。

自回归生成时,每个新 token 都要 attend 到所有之前的 token。为了避免每步重算前面所有 K/V,标准做法是把 K/V cache 在显存里,每步只算新 token 的 K/V 并 append。但 KV cache 的大小是:

$$
\text{KV cache} = 2 \times N \times L \times h \times d_k \times \text{bytes}
$$

LLaMA-2-70B 在 N=4096 上下文上:`2 × 4096 × 80 × 64 × 128 × 2 bytes = 10.7 GB / batch`。一个 80GB 的 A100 在 batch=4 时 cache 就占 43 GB,严重限制并发。

**Multi-Query Attention(MQA, Shazeer 2019)** 和 **Grouped-Query Attention(GQA, Ainslie 2023)** 给出的方案是 **共享 K/V 头**:

- **MHA**(原版):每个 head 有独立的 Q、K、V → `h` 套 K/V
- **MQA**:`h` 套 Q,但所有 head 共享 1 套 K/V → KV cache 降到 1/h
- **GQA**:把 `h` 个 head 分成 `g` 组,每组共享 1 套 K/V → KV cache 降到 g/h

MQA 太激进,质量掉点;GQA 是平衡选择,`g=8` 时质量几乎不掉、KV cache 降到 1/4。LLaMA 2 在 7B 上用 MHA、在 34B/70B 上用 GQA(`g=8`),Mistral / Mixtral / Qwen / Claude 3 也都用 GQA。

这是 attention 在头维度上的"稀疏化"——和 FlashAttention 在序列维度上的优化完全正交、可以同时用。**现代 LLM 推理 = FlashAttention + GQA**。

## 关键代码

完整 FlashAttention CUDA kernel 几千行(在 `flash-attn` 仓库),核心算法用 Python 写大概是这样:

```python
import torch

def flash_attention(Q, K, V, block_size=128):
    """简化版 FlashAttention(教学用,真实实现要 CUDA kernel)"""
    B, h, N, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    # 输出和 softmax 统计量
    O = torch.zeros_like(Q)
    L = torch.zeros(B, h, N, device=Q.device)  # row sum of exp
    M = torch.full((B, h, N), -float('inf'), device=Q.device)  # row max

    Br = Bc = block_size
    Tr = (N + Br - 1) // Br  # Q 的 block 数
    Tc = (N + Bc - 1) // Bc  # K 的 block 数

    for i in range(Tr):
        Q_i = Q[:, :, i*Br:(i+1)*Br]                          # [B, h, Br, d]
        O_i = O[:, :, i*Br:(i+1)*Br]
        l_i = L[:, :, i*Br:(i+1)*Br]                          # [B, h, Br]
        m_i = M[:, :, i*Br:(i+1)*Br]

        for j in range(Tc):
            K_j = K[:, :, j*Bc:(j+1)*Bc]
            V_j = V[:, :, j*Bc:(j+1)*Bc]

            # 在 SRAM 里算这个 block 的 partial attention
            S_ij = (Q_i @ K_j.transpose(-2, -1)) * scale     # [B, h, Br, Bc]
            m_ij = S_ij.max(dim=-1).values                    # [B, h, Br]
            P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))       # 数值稳定 exp
            l_ij = P_ij.sum(dim=-1)

            # online softmax 更新
            m_new = torch.maximum(m_i, m_ij)
            alpha = torch.exp(m_i - m_new)                    # 旧统计量的 rescale
            beta = torch.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * l_ij

            # 更新输出 — 用 rescale 把 O_i 调到与 m_new 一致的尺度
            O_i = (l_i.unsqueeze(-1) * alpha.unsqueeze(-1) * O_i +
                   beta.unsqueeze(-1) * (P_ij @ V_j)) / l_new.unsqueeze(-1)
            m_i, l_i = m_new, l_new

        O[:, :, i*Br:(i+1)*Br] = O_i
        L[:, :, i*Br:(i+1)*Br] = l_i
        M[:, :, i*Br:(i+1)*Br] = m_i

    return O
```

这个 Python 实现**比朴素 PyTorch attention 慢很多**——因为它没在 SRAM 里跑,每个 block 仍要走 HBM。真正的加速来自 CUDA kernel——把内部双重循环融合到一个 kernel 里,Q-block / K-block 直接从 HBM 加载到 SRAM,中间所有 `S_ij, P_ij` 都不写回 HBM。教学版只是为了说明"online softmax + 分块计算 = exact attention"的算法等价性。

实际用 FlashAttention 的代码非常简单:

```python
# PyTorch 2.0+ 内置
import torch.nn.functional as F
O = F.scaled_dot_product_attention(Q, K, V)  # 后端自动选 FlashAttention

# 直接用 flash-attn 库(更多控制)
from flash_attn import flash_attn_func
O = flash_attn_func(Q, K, V, causal=True)
```

一行替代,**所有 attention 自动加速**。这是 FlashAttention 工业影响力的关键——零开发成本接入。

## 影响 / 后续

FlashAttention 在 2022 年发表后两年内成为深度学习基础设施的一部分:

- **PyTorch 2.0**(2023 March)把 FlashAttention 集成进 `F.scaled_dot_product_attention`,所有 PyTorch attention 默认走它
- **HuggingFace Transformers** 添加 `attn_implementation="flash_attention_2"` flag,一行启用
- **vLLM / TGI / TensorRT-LLM** 等推理框架全部内置 FlashAttention 作为 default backend
- **xFormers**(Meta)、**FlashAttention-2/3** 持续优化迭代,跟进新硬件(A100 → H100 → B200)

FlashAttention 的更深影响是**重新定义了 attention 优化的方向**——从"算法层面减少 FLOPs"(稀疏 attention)转向"系统层面优化内存访问"。这一思路被推广到很多其他算子:

- **Linear Attention 的工业实现** — 类似的分块 + recompute 思路被用到 Linear Attention(RWKV、Mamba 等),让 O(N) 算法的工程效率匹配 FlashAttention
- **Flash Decoding**(2023) — 推理特定优化,把 KV cache 划分到多个 SM 并行
- **PagedAttention**(vLLM)— KV cache 的分页管理,和 FlashAttention 互补
- **Triton 编程模型** — Tri Dao 把 FlashAttention 用 Triton 重写后,Triton 成为写自定义 GPU kernel 的主流选择(替代手写 CUDA)

更广义地说,FlashAttention 是**深度学习从"算法优先"转向"硬件协同设计"的标志事件**。在它之前,论文里几乎只关心 FLOPs;在它之后,内存带宽、SRAM 利用率、kernel fusion 这些系统指标进入了主流讨论。今天的 LLM 训练已经几乎完全是"系统问题"——FlashAttention、3D parallelism、FSDP、ZeRO 这些工程优化的影响超过了算法本身。

→ [01-transformer.md](01-transformer.md) · 父结构;FlashAttention 是 attention 的系统级实现
→ [03-sparse-attention.md](03-sparse-attention.md) · 算法路线;FlashAttention 部分淘汰了它在 64K 以下的场景
→ [04-rope.md](04-rope.md) · 位置编码,与 FlashAttention 完全正交
→ [../07-gpt-scaling/](../07-gpt-scaling/) · 现代 LLM 的训练和推理都跑在 FlashAttention + GQA 上
→ [../13-moe-efficient/](../13-moe-efficient/) · 高效化的系统路线,与 FlashAttention 同源
