# MoE 与高效大模型

> **把"参数规模"和"激活计算"解耦——MoE 让模型有 T 级参数但每 token 只激活百分之几,在固定算力下把参数池放大十倍。**

## 一句话定位

这家族解决的是 dense LLM 的根本扩展瓶颈——**参数规模和单 token 算力锁死,要让模型变大就必须等比例多算**。GPT-3 175B 训练成本是 GPT-2 1.5B 的近 600 倍,这条路在 100B-1T 之间逼近物理 / 经济极限。2017 年 Shazeer 等人的 **"Outrageously Large Neural Networks"** 给出 LLM 时代之前的答案:**Sparsely-Gated Mixture-of-Experts**——把一个大 FFN 拆成 N 个 expert,每 token 只用一个 gate 选 top-K 个 expert,参数池可以做到 1370 亿但单步算力不变。2021 年 **Switch Transformer**(Google)把 MoE 推到 Transformer 主流——简化为 top-1 gating、加 load balancing loss,做出 1.6 T 参数模型,与 T5-XXL(11B dense)算力相当但质量大幅领先。2024 年初 **Mixtral 8×7B**(Mistral)第一次把 MoE 完全开源,8 个 7B expert + top-2 gating,46.7B 总参数但每 token 只激活 13B,推理质量超过 LLaMA-2-70B。2024 年底 **DeepSeek-V3**(671B 总参 / 37B 激活)集成 fine-grained experts + shared experts + auxiliary-loss-free balancing 等十几项创新,开源 MoE 旗舰首次追上 GPT-4 级闭源模型,也是 [DeepSeek-R1](../15-reasoning-o1-r1/04-deepseek-r1.md) 的 base 模型。这家族要回答的问题是:**如何用稀疏激活把 LLM 参数规模从 100B 推到 T 级,同时保持训练 / 推理算力可控**。

## 概念本身

MoE 的核心思路是**条件计算(conditional computation)**——不同 token 走不同的子网络,而不是所有 token 都走同一个庞大网络。

### 标准 dense Transformer FFN

```
Input token x (dim=d)
    ↓
FFN: x → W₁(d × 4d) → ReLU → W₂(4d × d) → output
```

所有 token 都用同一个 (W₁, W₂),参数 ~ 8d² 每层。

### MoE FFN

```
Input token x
    ↓
Gate(x): softmax 选 top-K experts(N 个候选)
    ↓
top-K experts 分别计算:Eᵢ(x) = W₂ᵢ · ReLU(W₁ᵢ · x)
    ↓
Output = Σ gateᵢ(x) · Eᵢ(x)
```

- N = 8(或 64, 256...)个 expert,每个 expert 是一个独立 FFN
- top-K 通常是 1 或 2
- 总参数 ~ N × 8d²(N 倍),激活参数 ~ K × 8d²(K 倍)

**关键解耦**:总参数(知识容量) vs 激活参数(单 token 算力)分离。Mixtral 8×7B 总参 46.7B,激活 13B——质量像 ~50B,推理速度像 13B。

### 几个核心设计点

**1. Gating 函数** —— 怎么选 expert?Top-K with softmax 是主流。Switch Transformer 用 top-1(更简单更快),Mixtral / DeepSeek-V3 用 top-2 / top-8(平衡质量与算力)

**2. Load Balancing** —— 不加约束的话 gate 会塌缩到几个"明星 expert",其他闲置。要加 **auxiliary loss** 鼓励 token 均匀分布到 expert(Shazeer 2017 提出)。或用 **router z-loss / aux-free balancing**(DeepSeek-V3 的创新)

**3. Expert Parallelism** —— 训练时不同 expert 放不同 GPU,通过 all-to-all 通信交换 token。是 MoE 工程化的最大难点

**4. Capacity Factor** —— 每 expert 能接受多少 token 上限。低 → 训练快但有 token 被 drop,高 → 训练慢但质量好

**5. Expert Granularity** —— DeepSeek-V3 把每个 expert 切得更细 + 加 shared experts(所有 token 都走),细粒度专精 + 通用能力共享

### MoE vs Dense 的本质差异

| 维度 | Dense | MoE |
|------|------|------|
| 总参数 | N | k × N(k 是 expert 数) |
| 激活参数 | N | ~ N(K 个 expert 中 K=1-2) |
| 训练算力 | O(N) | O(N)(但通信开销大) |
| 推理算力 | O(N) | O(N)(单 token) |
| 显存 | O(N) | **O(k × N)**(MoE 显存大!) |
| 质量 / 训练 token | 1× | 1.5-2×(同算力 budget) |

MoE 不是"免费的午餐"——总参数大意味着显存大,推理时所有 expert 都要常驻 GPU(虽然每次只用一两个)。MoE 的优势是**在固定训练算力下质量更高 / 在固定推理算力下质量更高**,代价是显存和工程复杂度。

围绕 MoE 演化的几条主线:

- **Gating 简化**:Shazeer 2017 top-K → Switch 2021 top-1 → Mixtral 2024 top-2(平衡复杂度与质量)
- **Expert 粒度**:粗粒度(8 大 expert)→ 细粒度(256 小 expert,DeepSeek-V3)
- **Load balancing**:auxiliary loss → router z-loss → aux-loss-free(V3)
- **训练稳定性**:Megatron-MoE、DeepSpeed-MoE 等系统级优化

理解 MoE 家族 = 理解 LLM 怎么突破 dense 模型在 100B-1T 之间的算力天花板。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2017 | **Outrageously Large NN** | Sparsely-gated MoE on LSTM,top-K gate + auxiliary loss + expert parallelism;1370 亿参数语言模型,30× dense 算力效率 | Dense 网络扩参必须等比例多算,无法把参数池做大 |
| 2021 | **Switch Transformer** | MoE 移植到 Transformer,top-1 gating 简化 + load balancing,1.6T 参数;Google T5-MoE 系列起源 | Shazeer 2017 用在 LSTM 上,Transformer 时代 MoE 工程化还没完成 |
| 2024 | **Mixtral 8×7B** | 第一个完全开源 MoE 旗舰,8 expert × 7B + top-2 gating,46.7B 总参 / 13B 激活;开源社区首次拿到生产级 MoE 模型 | 之前 MoE 模型都是 Google 闭源(Switch / GLaM / GShard),开源社区无 MoE 可用 |
| 2024 | **DeepSeek-V3** | 671B / 37B 激活,fine-grained experts + shared experts + aux-loss-free balancing + MTP,首次让开源 MoE 追上 GPT-4 级闭源模型,也是 R1 的 base | Mixtral 验证了开源 MoE 可行,但参数和质量离闭源旗舰仍差一代 |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— MoE 替换的是 Transformer 的 FFN 层
- [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) —— MoE 改变了 scaling law 的参数 / 算力比
- [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) —— GPT-4 据传也是 MoE 架构

**通向哪些家族:**
- [../15-reasoning-o1-r1/04-deepseek-r1.md](../15-reasoning-o1-r1/04-deepseek-r1.md) —— R1 直接基于 DeepSeek-V3 (MoE) 训练
- [../11-peft-lora/](../11-peft-lora/) —— MoE 微调成本高,PEFT 在 MoE 上的应用是新方向
- [../14-rag-agent/](../14-rag-agent/) —— MoE 大模型作为 agent / RAG 底座
