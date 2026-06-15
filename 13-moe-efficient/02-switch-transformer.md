---
name: "Switch Transformer"
year: 2021
family: "13-moe-efficient"
order: 2
paper: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
authors: ["William Fedus", "Barret Zoph", "Noam Shazeer"]
key_idea: "把 MoE 移植到 Transformer + 简化为 top-1 gating(每 token 只走一个 expert,代替 Shazeer top-K),配 load balancing loss 和 selective precision;首次做到 1.6T 参数模型,T5-XXL 4× 加速同质量"
---

## 前作进展

2020 年中,Transformer 时代 LLM 扩参已逼近极限:

- **T5-XXL**(2020,Google)11B dense,训练用 1024 块 TPU
- **GPT-3**(2020,OpenAI)175B,训练成本估 $4.6M
- **Megatron-LM**(NVIDIA)用 model parallelism 把 dense 推到 530B,但训练成本指数增长

要做到 1T 参数 dense 模型几乎不可能——算力 / 显存 / 通信都撞墙。

[Shazeer 2017](01-sparsely-gated-moe.md) 在 LSTM 上证明了 MoE 可以解耦"参数容量"和"激活算力",但有几个 Transformer 时代未解的工程问题:

**1. MoE 还没移植到 Transformer** —— Shazeer 2017 是 LSTM-MoE,Transformer 的 FFN 层结构不同,需要重新设计

**2. Top-K gating 复杂** —— Shazeer 用 K=4,实现复杂(每 token 路由到 4 个 expert,通信和计算都复杂)。能不能 K=1?

**3. Load balancing 不稳** —— 经常出现 expert 塌缩,训练不收敛

**4. 大规模分布式难** —— 1024+ 设备上跑 MoE,通信和精度问题没有成熟方案

GShard(Lepikhin 2020,Google)第一次把 MoE 用在 Transformer(600B 参数翻译模型),但用的还是 top-K(K=2),工程复杂。

Fedus 等人(Google Brain,2021 年 1 月)的 Switch Transformer 给出工程上的"终极简化"——**top-1 gating + 一系列稳定性 trick**,把 MoE 从研究 demo 推到 1.6T 参数生产模型。

## 核心思想:Top-1 Gating

Switch Transformer 的核心简化:**每 token 只去一个 expert**(top-1)。

### Top-1 Gating 公式

输入 token $x$:

$$
G(x) = \text{Softmax}(x \cdot W_g) \quad \text{(N 个 expert)} \\
i^* = \arg\max_i G(x)_i \\
y = G(x)_{i^*} \cdot E_{i^*}(x)
$$

只算一个 expert,gate 权重直接乘到该 expert 的输出上。

### 为什么 Top-1 行得通?

Shazeer 2017 用 K=4 是因为单 expert 表达力不够。Switch Transformer 的关键洞察:**在 Transformer 里 expert 数量足够多 + 训练数据足够大,top-1 就够好**。论文里 expert 数从 8 到 2048 不等,质量随 expert 数提升。

Top-1 的优势:

- **路由计算量减半** —— K=4 时一个 token 要算 4 个 expert 输出 + 加权;K=1 只算 1 个
- **all-to-all 通信减半** —— K 倍 token 量减到 1 倍
- **实现简单** —— 不用 weighted sum,直接 routing

### Load Balancing Loss(优化版)

Switch Transformer 用一个更简洁的 load balancing loss:

$$
L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i
$$

- $f_i$ —— 分给 expert $i$ 的 token 比例(0~1)
- $P_i$ —— 平均 gate 概率分给 expert $i$ 的(0~1)
- $\alpha = 0.01$ —— 权重系数

直观理解:如果某 expert 又被选中多($f_i$ 大)又拿到高概率($P_i$ 大),loss 就大,鼓励均衡。

### Capacity Factor

每 expert 有 token capacity 上限:

$$
\text{capacity} = \text{capacity\_factor} \cdot \frac{\text{tokens per batch}}{N}
$$

- capacity_factor = 1.0 表示"理想均衡时刚好装下"
- 实际用 1.25-2.0,留缓冲

超出 capacity 的 token 被 **drop**(直接跳过 MoE 层,通过 residual connection 传下去)。这是质量与算力的工程妥协。

### Selective Precision

MoE 训练经常因为 softmax + log 计算精度问题不稳定。Switch Transformer 提出 **selective precision**:大部分计算用 bfloat16(省显存),但 router 的 softmax 和 log 用 float32(精度高)。这一 trick 让 MoE 训练稳定性大幅提升。

### Differentiable Load Balancing

Switch Transformer 还设计了 differentiable expert assignment:即使 top-1 routing 是离散的,也能反向传梯度——通过把"被选中的 expert 的 gate 权重"乘到输出上,梯度可以经过 gate 反传到 $W_g$。

## 关键代码

Switch Transformer FFN 层简化实现:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=128, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, seq, d_model)
        b, s, d = x.shape
        x_flat = x.view(b * s, d)
        T = b * s

        # 1. Gating(selective precision:gate 用 float32)
        gate_logits = self.gate(x_flat.float())  # (T, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_prob, top_idx = gate_probs.max(dim=-1)  # (T,), (T,)

        # 2. Capacity check
        capacity = int(T * self.capacity_factor / self.num_experts)
        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (top_idx == e)
            tokens_for_e = mask.nonzero(as_tuple=True)[0]
            # drop 超出 capacity 的 token
            if tokens_for_e.numel() > capacity:
                tokens_for_e = tokens_for_e[:capacity]
            if tokens_for_e.numel() > 0:
                expert_out = self.experts[e](x_flat[tokens_for_e])
                output[tokens_for_e] = top_prob[tokens_for_e, None] * expert_out

        # 3. Load balancing loss
        # f_i = fraction of tokens assigned to expert i
        f = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            f[e] = (top_idx == e).float().mean()
        P = gate_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        return output.view(b, s, d), aux_loss * 0.01
```

实际工业实现(Google T5X / Megatron-MoE)还要处理:

- **all-to-all 通信**:expert 分布在多设备
- **token dropping 的 backwards** —— 被 drop 的 token 不参与 loss
- **expert parallelism**:expert 维度 sharding

## 性能数据

Switch Transformer 在 C4 语言建模和 GLUE / SuperGLUE 上的成绩:

| Model | 总参数 | 激活参数 | C4 perplexity | 训练步数到 T5-Base 质量 |
|------|------|------|------|------|
| T5-Base(dense) | 220M | 220M | 5.85 | 1.0× |
| T5-Large | 770M | 770M | 5.24 | 2.0× |
| T5-XXL | 11B | 11B | 4.65 | 60× |
| **Switch-Base**(N=128) | 7B | 0.22B | 5.32 | **0.5×**(2× 提速) |
| **Switch-Large**(N=128) | 26B | 0.77B | 4.87 | 0.4× |
| **Switch-XXL**(N=64) | 395B | 11B | 4.41 | 0.25×(**4× 提速**) |
| **Switch-C**(N=2048) | **1.57T** | 11B | 4.05 | - |

关键观察:

- **Switch-XXL 用 T5-XXL 同算力达到更好质量** —— 395B 总参 / 11B 激活,perplexity 4.65 → 4.41
- **同质量下提速 4-7×** —— 给定 target perplexity,Switch 训练步数远少于 dense
- **1.57T 参数** —— 首个公开的 trillion-parameter 模型,虽然激活只有 11B

下游 fine-tune 任务(SuperGLUE, GLUE, SQuAD):

| Model | SuperGLUE | GLUE | SQuAD F1 |
|------|------|------|------|
| T5-Base | 76.2 | 84.0 | 83.6 |
| Switch-Base(7B) | **77.5** | **85.0** | **85.4** |
| T5-Large | 82.9 | 87.7 | 87.5 |
| Switch-Large(26B) | **84.7** | **88.0** | **89.2** |

Switch-Base(7B 总参,激活 0.22B)在 SuperGLUE 上击败 T5-Base(220M dense),用同样的激活算力。

## 影响 / 后续

Switch Transformer 在 LLM 历史的位置:**MoE 第一次在 Transformer 上工程化成熟,直接催生现代 MoE 旗舰**。

**1. Top-1 gating 成主流** —— 之后 GLaM(2022)、ST-MoE(2022)等基本都用 top-1 或 top-2。Top-K=4 的 Shazeer 风格被工程化简化淘汰

**2. T5-MoE 成 Google 内部基础设施** —— Switch 之后 Google 把所有大型 NLP 模型(包括 PaLM 部分变体)都基于 MoE。GLaM(1.2T) / Gemini-1.5 据传都是 MoE 谱系

**3. Load balancing 框架标准化** —— Switch 的 $f \cdot P$ 形式 loss 成为后续所有 MoE 的默认。直到 DeepSeek-V3 才提出 aux-free 替代方案

**4. Capacity factor + token dropping** —— 工程上"宁可 drop token 也不让某 expert 撑爆"的策略成共识。后来 expert choice routing(Zhou 2022)反过来——让 expert 选 token——是另一思路

**5. Selective precision** —— "router 用高精度,其余用低精度"成 MoE 训练标配。bf16 + fp32 router 是今天 Megatron-MoE / DeepSpeed-MoE 默认配置

**6. 开源浪潮的催化剂** —— Switch 是闭源的,但论文公开后催生了开源复现工作。FairScale(Facebook)、DeepSpeed-MoE(Microsoft)、Megatron-MoE(NVIDIA)都基于 Switch 设计。最终 2024 年 Mistral 开源 Mixtral,让 Switch 的工程范式完全公开化

Switch Transformer 留下的开放问题:

- **Token dropping 的质量损失** —— 被 drop 的 token 完全跳过 MoE 层,损失多少质量没量化 → Expert Choice 路由部分解决
- **推理时所有 expert 都要常驻显存** —— 1.57T 模型推理需要海量显存。MoE 推理优化是开放工程问题
- **Expert 数量 vs 质量** —— 给定算力 budget,expert 数越多越好吗?后续工作发现存在 sweet spot
- **多模态 / 跨任务 MoE** —— 不同 expert 是否能学到不同"能力"(数学专家、代码专家)?MoE-LLaVA 等工作探索这一方向

→ [03-mixtral.md](03-mixtral.md) · 开源版 Switch,Mistral 用 top-2 而非 top-1
→ [04-deepseek-v3.md](04-deepseek-v3.md) · MoE 集大成,fine-grained experts + aux-loss-free
→ [01-sparsely-gated-moe.md](01-sparsely-gated-moe.md) · 祖师爷,Switch 是其 Transformer 化简化版
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · Switch 替换的是 Transformer 的 FFN
→ [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) · MoE 改变了 scaling law 的 N/C 关系
