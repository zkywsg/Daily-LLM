---
name: "DeepSeek-V3"
year: 2024
family: "13-moe-efficient"
order: 4
paper: "DeepSeek-V3 Technical Report"
authors: ["DeepSeek-AI"]
key_idea: "671B 总参 / 37B 激活的开源 MoE 旗舰,集成 fine-grained experts(256 细粒度 expert)+ shared experts + aux-loss-free load balancing + MTP(Multi-Token Prediction)等十余项创新;首次让开源 MoE 追上 GPT-4 级闭源模型,也是 DeepSeek-R1 的 base"
---

## 前作进展

2024 年中开源 MoE 状况:

- **[Mixtral 8×7B](03-mixtral.md)** 验证了开源 MoE 可行,但 46.7B 总参 / 13B 激活,质量与 GPT-4 仍有明显差距
- **DeepSeek-V2**(2024.5)236B / 21B 激活,fine-grained experts 初版本,效果接近 LLaMA-3-70B 但未达 GPT-4
- **闭源 MoE** 不断推进(GPT-4o, Claude-3.5, Gemini-1.5),细节不公开

开源 vs 闭源差距:GPT-4 (2023.3) → 开源追到 GPT-4 (2024.中) 用了 18 个月。但 GPT-4 之上还有 GPT-4o / Claude 3.5 Sonnet,差距仍在。

DeepSeek 团队在 V2 之后用 6 个月时间集成了过去几年 MoE 研究的几乎全部工程经验,2024 年 12 月 26 日发布 **DeepSeek-V3** 技术报告(685B 总参,论文里写 671B 是去掉 shared 部分)。发布后引爆开源 LLM 圈:

- **MMLU 88.5, MATH 90.2, HumanEval 82.6** —— 全面持平甚至略超 GPT-4o / Claude-3.5-Sonnet
- **训练成本仅 $5.6M** —— GPT-4 训练估 $100M,V3 用 1/20 成本达到同等质量
- **完全开源**(权重 + 详细技术报告 + 推理代码)
- **2025 年 1 月在 V3 上做 RL 训出 [R1](../15-reasoning-o1-r1/04-deepseek-r1.md)** —— V3 是 R1 的 base,本家族与 reasoning 家族直接连通

DeepSeek-V3 不是单一创新,而是 **MoE 集大成者**——集成 fine-grained experts、shared experts、aux-loss-free balancing、MTP、FP8 训练、MLA attention 等十余项工程优化,每一项都把质量推一小步,合起来让开源首次超过闭源旗舰。

## 核心思想:Fine-Grained MoE + Aux-Free Balancing

V3 的工程创新太多,选最关键的几个讲。

### 1. Fine-Grained Experts(细粒度专家)

V2/V3 的核心架构创新:**把 expert 切得更细,数量更多**。

| | Mixtral 8×7B | DeepSeek-V3 |
|------|------|------|
| Expert 数(每层)| 8 | **256**(routed)+ 1(shared) |
| Top-K | 2 | **8** |
| 每 expert 大小 | ~5.5B | ~0.5B(更小) |
| 激活 expert 总参数 | ~11B | ~4B(8 个 small) |

直观对比:Mixtral 像"8 个大全能选手选 2 个",V3 像"256 个专才选 8 个"。

为什么细粒度更好?论文给出两个解释:

- **更高的专精度**:细 expert 更容易 specialize 到特定模式(代码风格、数学步骤、等)
- **更精细的组合空间**:C(256, 8) 远多于 C(8, 2),token 路由的组合多样性巨大

### 2. Shared Experts(共享专家)

V3 引入 **shared expert**——一个所有 token 都走的通用 expert,与 routed expert(top-K 选 8 个)并行。

```
Output = SharedExpert(x) + Σ_{i ∈ top-8} G(x)_i · RoutedExpert_i(x)
```

直觉:每个 token 都有通用基础能力(语法、常识),不需要每次都从 256 个 expert 里选一个负责通用能力。Shared expert 承担通用,routed expert 负责特化,这样 routed expert 可以更专精。

### 3. Auxiliary-Loss-Free Load Balancing

传统 MoE(Switch / Mixtral)用 auxiliary loss 强制 expert 负载均衡。但 aux loss 会**与主任务 loss 冲突**——为了 balance,gate 不能完全自由选最优 expert。

V3 创新:**用 bias 替代 aux loss**。每 expert 有一个 learnable bias $b_i$:

$$
\text{score}_i = \text{Sigmoid}(x \cdot W_g)_i + b_i
$$

训练时监控每 expert 的 load(被选中频率)。如果某 expert 过载,把它的 $b_i$ 调小(让它更少被选);如果欠载,调大。这个 bias 调整 **不参与梯度**,只是在线动态平衡。

效果:**主任务 loss 不被 aux loss 干扰,但 expert 仍然均衡**。这是 V3 比 Mixtral 训练更稳定的关键。

### 4. MLA(Multi-head Latent Attention)

DeepSeek-V2 提出、V3 延续的 attention 优化。把 KV cache 压缩到低维 latent 空间:

```
传统 MHA: K_h, V_h ∈ R^{d_head} 每个 head 独立存
MLA:   把 K, V 压缩到 d_latent(远小于 ∑ d_head)的 latent
```

V3 的 MLA 让 KV cache 大小降到传统 MHA 的 ~1/4,长上下文推理速度大幅提升。是 V3 能上 128K 上下文的关键。

### 5. MTP(Multi-Token Prediction)

V3 训练时引入 MTP——每个位置不只预测下一个 token,而是预测**未来 D 个 token**(论文用 D=2)。多 token 预测让信号更密集,提升训练效率:

```
传统:   loss = -log p(x_{i+1} | x_{≤i})
MTP:    loss = Σ_{d=1}^D -log p(x_{i+d} | x_{≤i+d-1}, predicted x_{i+1..i+d-1})
```

MTP 还能用作**推理加速**——预测下两个 token,如果第二个对就跳一步(speculative decoding 的训练时版本)。

### 6. FP8 训练

V3 是第一个公开宣称"全流程 FP8 训练"的开源大模型。GEMM 算子用 FP8,LayerNorm / activations 用 BF16。FP8 比 BF16 快 ~2×、省 ~50% 显存,但精度更敏感——V3 通过 fine-grained quantization scaling 解决精度问题。

FP8 训练让 V3 在 2048 块 H800 上用 2.79M GPU-hours 训完(GPT-4 估算 100M+ GPU-hours)。**训练效率提升 10×+** 是 V3 总成本仅 $5.6M 的关键。

## 关键代码

V3 MoE 层简化结构(基于 DeepSeek 官方代码):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekV3MoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size,
                 num_routed_experts=256, num_shared_experts=1, top_k=8):
        super().__init__()
        self.num_routed = num_routed_experts
        self.top_k = top_k
        # Router(无 aux loss,用 learnable bias)
        self.gate = nn.Linear(hidden_size, num_routed_experts, bias=False)
        # bias 用于 aux-free balancing,不参与 grad
        self.register_buffer("expert_bias",
                             torch.zeros(num_routed_experts))
        # Routed experts(每个比 Mixtral 的小很多)
        small_intermediate = intermediate_size // 8  # fine-grained
        self.routed_experts = nn.ModuleList([
            SwiGLUFFN(hidden_size, small_intermediate)
            for _ in range(num_routed_experts)
        ])
        # Shared experts(所有 token 都走)
        self.shared_experts = nn.ModuleList([
            SwiGLUFFN(hidden_size, small_intermediate)
            for _ in range(num_shared_experts)
        ])

    def forward(self, x):
        b, s, h = x.shape
        x_flat = x.view(-1, h)

        # 1. Routing with aux-free bias
        logits = self.gate(x_flat).sigmoid()  # V3 用 sigmoid 而非 softmax
        # 加 expert bias(动态调,不参与梯度)
        scores = logits + self.expert_bias.detach()
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        # 归一化 routing weights(用原始 logits,不含 bias)
        routing_weights = logits.gather(-1, topk_idx)
        routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)

        # 2. Shared experts(所有 token 都过)
        shared_out = sum(e(x_flat) for e in self.shared_experts)

        # 3. Routed experts(top-8)
        routed_out = torch.zeros_like(x_flat)
        for e_idx in range(self.num_routed):
            mask = (topk_idx == e_idx).any(-1)
            if not mask.any():
                continue
            # 拿到选中此 expert 的 token 和对应权重
            token_mask = mask.nonzero(as_tuple=True)[0]
            # 找到这些 token 在 topk_idx 里 e_idx 的位置
            slot = (topk_idx[token_mask] == e_idx).float().argmax(-1)
            w = routing_weights[token_mask, slot:slot+1]
            out = self.routed_experts[e_idx](x_flat[token_mask])
            routed_out[token_mask] += w * out

        # 4. Update expert_bias(训练时,根据 load 动态调)
        if self.training:
            with torch.no_grad():
                load = torch.zeros(self.num_routed, device=x.device)
                for k in range(self.top_k):
                    load.scatter_add_(0, topk_idx[:, k], torch.ones_like(topk_idx[:, k], dtype=load.dtype))
                target = load.mean()
                self.expert_bias -= 0.001 * (load - target)  # 简化版

        return (shared_out + routed_out).view(b, s, h)
```

完整 V3 还要处理 MLA attention、MTP loss、FP8 quantization 等模块,代码量数千行。

## 性能数据

V3 在主流 LLM benchmark 上的成绩(技术报告 Table 4):

| Model | MMLU | MMLU-Pro | DROP F1 | MATH | HumanEval | LiveCodeBench |
|------|------|------|------|------|------|------|
| LLaMA-3.1-405B | 88.6 | 73.3 | 84.8 | 73.8 | 89.0 | 28.4 |
| GPT-4o(2024-08) | 87.2 | 72.6 | 83.7 | 76.6 | 91.0 | 33.4 |
| Claude-3.5-Sonnet(2024-10) | 88.3 | 78.0 | **88.3** | 78.3 | 92.0 | 36.3 |
| Qwen-2.5-72B | 85.0 | 71.6 | 76.7 | 80.0 | 86.6 | 31.4 |
| **DeepSeek-V3** | **88.5** | **75.9** | 84.0 | **90.2** | 82.6 | **40.5** |

关键观察:

- **MATH 90.2 超 Claude-3.5-Sonnet 78.3** —— 数学能力开源第一次反超闭源旗舰
- **MMLU 88.5 与 Claude-3.5-Sonnet 持平**(88.3)
- **LiveCodeBench 40.5 超 Claude-3.5-Sonnet 36.3** —— 代码能力开源第一次超闭源
- **HumanEval 略低于闭源**(82.6 vs 91-92)—— 但 LiveCodeBench(更真实)反超

效率对比:

| Model | 总参 | 激活 | 训练成本(估) | 推理成本(per 1M tokens) |
|------|------|------|------|------|
| LLaMA-3.1-405B | 405B | 405B | ~$60M | ~$5-10 |
| GPT-4o | ? | ? | ~$100M+ | $2.5-10 |
| Claude-3.5-Sonnet | ? | ? | ~$100M+ | $3-15 |
| **DeepSeek-V3** | **671B** | **37B** | **$5.6M** | **$0.27-1.1** |

V3 训练成本仅闭源的 ~5%,推理成本仅 ~10%。这是 2024 年 LLM 圈最震撼的效率数据。

## 影响 / 后续

DeepSeek-V3 在 LLM 历史的位置:**开源 LLM 第一次全面追平闭源旗舰,改变了"必须大算力才能训前沿模型"的认知**。

**1. 引爆开源 LLM 信心** —— V3 之前社区普遍认为开源至少落后闭源 1-2 年。V3 + R1 把这一差距压缩到几个月,甚至在某些维度反超。Yann LeCun 在 V3 发布后公开评论"开源已不可阻挡"

**2. R1 的 base 模型** —— [DeepSeek-R1](../15-reasoning-o1-r1/04-deepseek-r1.md) 直接在 V3 上做 RL 训练。**没有 V3 就没有 R1**——V3 提供了强 base + 经济的训练成本,让 R1 能在 V3 之上加 RL 阶段而总成本可控

**3. Aux-free balancing 被广泛复用** —— Qwen 2.5 / GLM-4 等后续开源 MoE 普遍采用 V3 的 aux-free 方案。Switch / Mixtral 风格的 aux loss 在开源 MoE 里逐渐淘汰

**4. Fine-grained experts 成主流** —— V3 的 256 expert 路线影响后续工作。Qwen3-MoE(2025)、Hunyuan-MoE 等都跟进细粒度方案

**5. FP8 训练普及** —— V3 全 FP8 训练打破"BF16 才能稳定训大模型"的信念。NVIDIA H100 / H200 / B200 的 FP8 算力开始被严肃利用,2025 年新发布的大模型大多包含 FP8 训练路径

**6. 算力市场冲击** —— V3 / R1 的"5.6M 美元训前沿模型"数字让 NVIDIA 股价单日跌 17%。市场重新评估"超大规模算力是否必要",对 H100 等高端 GPU 的长期需求被质疑。这一冲击波延续到 2025 整个上半年

**7. 中国 LLM 实力被重新定位** —— V3 + R1 双发让 DeepSeek 进入与 OpenAI / Anthropic / Google 并列的"前沿四家"。中国 LLM 研究第一次在顶级水平上有持续输出

V3 留下的开放问题:

- **可复现性** —— V3 训练用到 2048 卡 H800,只有少数实验室能复现完整训练。社区的复现工作仍在进行
- **MoE 推理工程** —— 671B 总参意味着推理需要海量显存,如何让"普通研究者也能跑 V3"是 expert offload / quantization 的开放方向
- **Reasoning + MoE 融合优化** —— V3 → R1 的训练成功,但 reasoning 模型本身是否适合 MoE 还没充分研究
- **Expert specialization 真的存在吗** —— V3 的 fine-grained experts 是否真的"分工"还是只是"组合空间大"——理论解释仍开放

→ [03-mixtral.md](03-mixtral.md) · 父节点,V3 = Mixtral + 4 年工程演化
→ [02-switch-transformer.md](02-switch-transformer.md) · V3 仍继承 Switch 的 top-K + load balancing 框架
→ [01-sparsely-gated-moe.md](01-sparsely-gated-moe.md) · 祖师爷
→ [../15-reasoning-o1-r1/04-deepseek-r1.md](../15-reasoning-o1-r1/04-deepseek-r1.md) · V3 是 R1 的 base,直接下游
→ [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) · V3 改写了 N/C scaling 的实际数字
→ [../11-peft-lora/](../11-peft-lora/) · V3 这样巨型 MoE 的微调需要 PEFT 方法
