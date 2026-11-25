# 参数高效微调 (PEFT)

> **不动 base 模型权重,只训几百万到几千万参数,就让 7B-70B LLM 适配新任务——把 LLM 微调成本从十万美元降到几十美元。**

## 一句话定位

这家族解决的是 LLM 时代的一个工程瓶颈——**全参微调一个 70B 模型需要 ~1TB GPU 显存 + 多卡集群,中小团队 / 个人开发者完全没法做**。2019 年 Houlsby 等人的 **Adapter** 给出第一个答案:**冻结 base 模型,在每层 Transformer 插入小 bottleneck 模块**,只训 0.5-1% 参数就能达到全参微调 95% 的效果。2021 年 Li & Liang 的 **Prefix Tuning** 走另一条路:**只训 input 前的 soft prompt token**,base 模型完全冻结,推理时把 prompt 拼到输入前。2021 年 Hu 等人的 **LoRA**(Low-Rank Adaptation)给出 PEFT 时代的"工业标准答案":**把权重更新 ΔW 分解为低秩矩阵 BA**,只训 r=8 或 16 的小矩阵,推理时可以合并回原权重不增加延迟。LoRA 出来后 6 个月被开源社区全面采用,成为今天微调 LLaMA / Mistral / Qwen 等开源模型的事实标准。2023 年 Dettmers 等人的 **QLoRA** 把 base 模型量化到 4-bit + 用 LoRA 微调,**让 65B 模型能在单卡 24GB 消费级 GPU 上微调**,把 LLM 微调彻底"个人化"。这家族要回答的问题是:**LLM 时代如何让千万开发者都能定制大模型,而不必拥有大公司的算力**。

## 概念本身

PEFT(Parameter-Efficient Fine-Tuning)的核心是**冻结大部分参数,只训一小部分**。把 LLM 微调成本和资源拆开:

```
Full Fine-Tuning(全参微调):
  base model (70B) → 全部反传 → 优化器需要 ~1TB 显存

PEFT:
  base model (70B, frozen) → 只反传到小模块(几 MB - 几 GB)
  优化器只更新 0.01% - 1% 参数
```

### 几条主要技术路线

**1. Adapter Tuning(适配器路线)**

在每层 Transformer 插入 small bottleneck 模块:`x → down(d→r) → ReLU → up(r→d) → +residual`。base 冻结,只训 adapter 参数(每层 ~ 4dr 参数,通常 r=64,d=4096 → 1M 参数每层)。

代表:Houlsby Adapter(2019)、Pfeiffer Adapter、AdapterHub 生态

**2. Prompt-based Tuning(软提示路线)**

只训"input 前的可学习 embedding"(soft prompt token),base 完全冻结。模型不知道这些是"特殊 token",当成普通 token 处理。

代表:Prefix Tuning(Li & Liang 2021)、Prompt Tuning(Lester 2021)、P-Tuning v2

**3. LoRA(低秩适应路线)**

把权重更新 ΔW 用低秩矩阵分解:$W = W_0 + BA$,其中 $A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$,$r \ll d$(通常 r=8/16/32)。

代表:LoRA(Hu 2021)、QLoRA(Dettmers 2023)、DoRA、LoRA+

**4. Quantization + PEFT(量化路线)**

base 模型量化到 4-bit / 8-bit,只 LoRA 部分用 fp16/bf16 训练。极大降低显存。

代表:QLoRA(2023)、AWQ + LoRA、GPTQ + LoRA

### PEFT 各路线对比

| 方法 | 训练参数 | 推理延迟 | 训练显存(7B 模型) | 备注 |
|------|------|------|------|------|
| Full FT | 100% | 1× | ~80 GB | 黄金标准但贵 |
| Adapter | ~3% | **1.1×**(加层) | ~30 GB | 推理稍慢 |
| Prefix/Prompt | ~0.1% | **1.05×**(加长序列) | ~25 GB | 性能上限低 |
| **LoRA** | **~0.1-1%** | **1×**(可合并) | ~20 GB | 工业标配 |
| **QLoRA** | ~0.5% | 1× | **~5 GB** | 消费级硬件 |

### 几个核心设计点

- **PEFT 在哪些层 / 哪些位置**:LoRA 默认 attention 的 Q,V;adapter 默认 FFN 后
- **rank r 怎么选**:r=4-16 一般够,r=64+ 接近全参
- **学习率 vs Full FT**:PEFT 学习率通常比 Full FT 大 10-100×(参数少梯度幅值小)
- **多 task 切换**:LoRA / adapter 可以保存为独立"插件",base 共享。一个 base + 100 个 LoRA = 100 个微调模型

围绕 PEFT 演化的几条主线:

- **从修改架构到修改输入**:Adapter(加层)→ Prefix(加 token)→ LoRA(改 weight delta)
- **从可学习参数减少到训练成本减少**:Prompt Tuning(参数最少)→ LoRA(参数适中)→ QLoRA(显存最少)
- **从单 task 到多 task / 持续学习**:LoRA composition、AdaLoRA、LoRA Hub

理解 PEFT 家族 = 理解 LLM 时代如何把"千万美元微调"压缩到"几十美元微调",让 LLM 真正普惠。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2019 | **Adapter Tuning**(Houlsby) | 在 Transformer 每层插 small bottleneck 模块,base 冻结,只训 3% 参数达全参 96% 性能;PEFT 起源 | BERT-large 全参微调 GLUE 36 任务需要 36 份 model checkpoint,存储 / 切换成本爆炸 |
| 2021 | **Prefix Tuning** | 只训 input 前的 soft prompt token,base 完全冻结;0.1% 参数,extreme parameter efficiency | Adapter 仍需修改架构,推理慢;不能做完全冻结的"plug-and-play" |
| 2021 | **LoRA** | 把权重更新 ΔW = BA 低秩分解,只训 r=8 的小矩阵,推理时合并回原权重零延迟;PEFT 工业标准 | Adapter 推理加延迟、Prefix 上限低;需要"零开销 + 高质量"的 PEFT |
| 2023 | **QLoRA** | Base 模型量化到 4-bit + LoRA 微调,paged optimizer + double quantization;65B 模型可在单卡 24GB GPU 微调 | LoRA 仍需 fp16 加载 base,7B 要 14GB / 70B 要 140GB,消费级硬件做不动 |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— PEFT 改的都是 Transformer 的内部结构
- [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) —— Adapter / Prefix Tuning 最早在 BERT 上验证
- [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) —— LoRA / QLoRA 的主战场是 LLaMA / Mistral / Qwen 等开源 LLM

**通向哪些家族:**
- [../12-rlhf-alignment/](../12-rlhf-alignment/) —— RLHF / DPO 普遍用 LoRA 训练以降低成本
- [../13-moe-efficient/04-deepseek-v3.md](../13-moe-efficient/04-deepseek-v3.md) —— MoE 模型的 LoRA 微调是新研究方向
- [../15-reasoning-o1-r1/04-deepseek-r1.md](../15-reasoning-o1-r1/04-deepseek-r1.md) —— R1 蒸馏版本广泛用 LoRA 进一步微调
- [../14-rag-agent/](../14-rag-agent/) —— Agent / RAG 系统的"领域适配"通常通过 LoRA 实现
