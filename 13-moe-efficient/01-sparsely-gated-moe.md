---
name: "Outrageously Large Neural Networks (Sparsely-Gated MoE)"
year: 2017
family: "13-moe-efficient"
order: 1
paper: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
authors: ["Noam Shazeer", "Azalia Mirhoseini", "Krzysztof Maziarz", "Andy Davis", "Quoc Le", "Geoffrey Hinton", "Jeff Dean"]
key_idea: "在 LSTM 之间插入 sparsely-gated MoE 层:每 token 用 gate 选 top-K 个 expert(1370 亿参数中只激活几亿),配 auxiliary loss 防止 expert 塌缩;首次证明稀疏激活能突破 dense 模型的参数 / 算力锁死"
---

## 前作进展

2015-2017 年深度学习扩参的核心痛点:**参数规模和计算量绑死**。给 LSTM 加一层就要等比例多算,GPU 显存和训练时间随参数线性增长。

社区当时有两条尝试方向:

**1. Mixture of Experts(MoE)古典版** —— Jacobs 1991 提出过 MoE 思路:有几个"专家"网络,gating 网络决定用哪个。但这一思路只在小模型上验证过,从未在深度网络里 work——dense gating(所有 expert 都算)就回到了 dense 网络,没有省算力

**2. Conditional Computation 一线** —— Bengio 2013 提出"条件计算":根据输入动态激活不同子网络。但实现起来要么 gating 不可微(无法端到端训练),要么实际算力没省下来(GPU 不擅长 sparse computation)

Shazeer 等人(Google Brain,2017 年 1 月)的论文给出第一个**真正 work 的稀疏 MoE 实现**——既保持端到端可微,又真的省算力,在工程上跑通了 1370 亿参数 LSTM。论文题目"Outrageously Large"反映了当时的震撼:**2017 年 GPT 还没出来,1370 亿参数比当时所有公开模型大一个量级**。

这一工作虽然用在 LSTM 上,但奠定了后来所有 MoE 工作的核心组件——top-K gating、auxiliary loss、expert parallelism 全是 Shazeer 2017 提出的。**2021 年 Switch / 2024 年 Mixtral / 2024 年 V3 都是这篇论文的孙辈**。

## 核心思想:Sparsely-Gated MoE

整体架构(LSTM-MoE 堆叠):

```
  LSTM ──┐
         ↓
   [MoE Layer]  ← N=2048 experts,top-K=4
         ↓
  LSTM ──┐
         ↓
   [MoE Layer]
         ↓
   ...
```

每个 MoE Layer 替换一个 dense FFN,但有 N 个 expert(论文里 N=2048,每个 expert 是一个小 FFN),每 token 只用 top-K=4 个。

### Gating 网络

输入 token $x \in \mathbb{R}^d$,gating 计算:

$$
G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g + \epsilon, K))
$$

- $W_g \in \mathbb{R}^{d \times N}$ —— gating 矩阵,把 $x$ 投影到 N 个 expert 的 logits
- $\epsilon$ —— 训练时加噪(noisy top-K),促进探索
- $\text{TopK}(\cdot, K)$ —— 保留 K 个最大值,其余设为 $-\infty$
- Softmax 后 K 个保留位置非零,其余精确 0

**关键**:精确 0 让对应 expert 完全不计算(算力真省下来)。这是与 dense gating 的根本区别。

### MoE 输出

$$
y = \sum_{i=1}^N G(x)_i \cdot E_i(x)
$$

由于 $G(x)$ 只有 K 个非零分量,实际只计算 K 个 $E_i(x)$——其余 N-K 个 expert 完全跳过。

### Auxiliary Loss(论文核心创新)

不加约束时 gating 会塌缩——少数 expert 被大量选中,其余闲置。论文设计两个 auxiliary loss:

**1. Importance Loss** —— 鼓励每个 expert 被选中的总 gate 权重均衡:

$$
L_{\text{importance}} = w \cdot \text{CV}(\text{Importance}_i)^2, \quad \text{Importance}_i = \sum_{x \in \text{batch}} G(x)_i
$$

CV 是变异系数(std / mean),小 = 均衡。

**2. Load Loss** —— 鼓励每个 expert 被选中的 token 数均衡(防止某 expert 被过度激活):

$$
L_{\text{load}} = w \cdot \text{CV}(\text{Load}_i)^2
$$

总 loss = task loss + importance + load。这两个 auxiliary loss 成为后来所有 MoE 工作的标配。

### Expert Parallelism

工程实现的关键:**N=2048 个 expert 分布在多个 GPU 上**。Shazeer 2017 用 128 块 K40 GPU,每 GPU 装 16 个 expert。

训练流程:

1. 每 GPU 算 gating,决定哪些 token 去哪个 expert
2. **All-to-all 通信**:把 token 路由到对应 expert 所在的 GPU
3. expert 本地计算
4. 反向 all-to-all:把输出送回原 GPU

这是 MoE 训练最大的工程难点——all-to-all 是密集通信,网络带宽和延迟是瓶颈。Shazeer 团队花了大量工程把它跑通。

## 关键代码

简化版 Sparsely-Gated MoE 层(PyTorch):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # 每个 expert 是一个 2 层 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, seq, d_model)
        b, s, d = x.shape
        x_flat = x.view(b * s, d)

        # 1. Gating
        logits = self.gate(x_flat)  # (b*s, num_experts)
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.1  # noisy top-K
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (b*s, top_k)
        weights = F.softmax(topk_vals, dim=-1)

        # 2. 计算 top-K expert(简化:dense 实现,真实版用 all-to-all)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (topk_idx[:, k] == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += weights[mask, k:k+1] * expert_out

        # 3. Auxiliary loss(每 expert 的平均 gate 权重均衡)
        gate_probs = F.softmax(logits, dim=-1)  # (b*s, num_experts)
        importance = gate_probs.sum(0)  # (num_experts,)
        aux_loss = importance.var() / (importance.mean() ** 2 + 1e-8)

        return output.view(b, s, d), aux_loss
```

真实实现要处理:

- **token dropping**:每 expert 有 capacity 上限,超出的 token 被 drop 或重路由
- **all-to-all communication**:expert 分布在多 GPU
- **load balancing trick**:capacity factor / expert choice routing

## 性能数据

Shazeer 2017 在 Google 1 Billion Word Language Modeling Benchmark(LM1B)和 WMT EN-FR 翻译任务上验证:

| Model | 参数 | 激活参数 | LM1B Perplexity | 训练算力(同基线) |
|------|------|------|------|------|
| LSTM-2048 baseline | 0.2B | 0.2B | 67.5 | 1× |
| LSTM-Big | 1.4B | 1.4B | 39.8 | 7× |
| **MoE-32-experts** | 0.8B | 0.18B | 35.7 | **1×** |
| **MoE-512-experts** | 4.4B | 0.42B | 31.3 | 1.4× |
| **MoE-2048-experts** | 137B | 1.5B | **28.0** | 2.4× |

关键观察:

- **137B MoE 用 LSTM-Big 1/3 算力达到 30% 更好的 perplexity** —— 第一次证明稀疏激活的规模优势
- **expert 数越多质量越好** —— 但收益递减,需要找 sweet spot
- **激活参数远小于总参数** —— 137B 总参,1.5B 激活,稀疏度 ~1%

WMT EN-FR 翻译:

| Model | 总参数 | BLEU |
|------|------|------|
| GNMT(2016 SOTA) | ~250M | 39.92 |
| **MoE-2048** | 8.7B | **40.56** |

MoE 第一次在工业级翻译任务上超过 dense baseline。

## 影响 / 后续

Shazeer 2017 在 LLM 历史的位置:**奠定了所有现代 MoE 的核心范式,Shazeer 也成为这条路线最重要的研究者之一**。

**1. 所有现代 MoE 的祖师爷** —— Top-K gating、auxiliary loss、expert parallelism 三大核心组件全部首次出现在这篇论文里。[Switch Transformer](02-switch-transformer.md)、GShard、GLaM、Mixtral、DeepSeek-V3 都直接继承这套设计

**2. Shazeer 本人的影响力** —— 论文一作 Noam Shazeer 后来还是 Transformer(2017,二作)、PaLM、Switch Transformer 的核心作者,2021 年离开 Google 创办 Character.AI(2024 年被 Google 25 亿美金收购)。Shazeer 是 LLM 时代最重要的工程师之一

**3. Conditional Computation 路线被验证** —— 在此之前 conditional computation 在大模型上一直没成功,Shazeer 2017 给了第一个可信的案例,启动了整个稀疏激活研究方向

**4. Auxiliary loss 框架被广泛复用** —— 不只在 MoE,后续 Switch 的 load balancing loss、DeepSpeed-MoE 的 z-loss、DeepSeek-V3 的 aux-free balancing 都建立在这个框架上

**5. 现代分布式训练系统的奠基** —— Expert Parallelism 的工程实现催生了 Mesh-TensorFlow、Megatron-MoE、DeepSpeed-MoE、FairScale 等分布式系统库。MoE 训练复杂度推动了整个分布式训练生态

**6. 直接通向 Transformer 时代的 MoE** —— Shazeer 2017 之后的 4 年里,GShard(2020)、Switch(2021)、GLaM(2021)等工作把 MoE 从 LSTM 平移到 Transformer,这一系列工作是直接发展线

Shazeer 2017 留下的开放问题(后续工作填):

- **LSTM 序列性导致 expert 调度不均** —— Transformer 的并行性更适合 MoE → Switch / GShard 解决
- **top-K=4 太复杂** —— [Switch Transformer](02-switch-transformer.md) 简化到 top-1
- **expert 塌缩仍偶发** —— 需要更鲁棒的 load balancing → router z-loss / aux-loss-free
- **训练不稳定 / 难复现** —— 工程系统不够成熟 → DeepSpeed / Megatron 后续完善

→ [02-switch-transformer.md](02-switch-transformer.md) · 把 MoE 移植到 Transformer + 简化为 top-1
→ [03-mixtral.md](03-mixtral.md) · 开源 MoE 旗舰,把 Shazeer 2017 思路完整开源
→ [04-deepseek-v3.md](04-deepseek-v3.md) · 现代 MoE 集大成者,V3 = Shazeer 2017 + 7 年工程演化
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · Shazeer 同时也是 Transformer 二作
→ [../15-reasoning-o1-r1/](../15-reasoning-o1-r1/) · DeepSeek-V3 (本家族) 是 R1 的 base
