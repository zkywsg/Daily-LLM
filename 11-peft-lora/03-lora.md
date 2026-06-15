---
name: "LoRA"
year: 2021
family: "11-peft-lora"
order: 3
paper: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Yuanzhi Li", "Shean Wang", "Lu Wang", "Weizhu Chen"]
key_idea: "把权重更新 ΔW 分解为低秩矩阵 B·A(r 远小于 d),只训 BA 的 ~0.1% 参数;推理时 W = W₀ + BA 可合并回原权重,零额外延迟;PEFT 时代的工业标准"
---

## 前作进展

2021 年中,PEFT 路线已有两条主流:

**1. Adapter Tuning**([Houlsby 2019](01-adapter.md))—— 加 bottleneck 模块,3% 参数达 96% 性能。但**推理延迟 +5-10%**,因为 adapter 层必须在 forward 时算

**2. Prefix Tuning**([Li & Liang 2021](02-prefix-tuning.md))—— 加 soft prompt,0.1% 参数。但**序列变长**(attention 复杂度 O((n+m)²)),且**只在大模型上稳定**

两条路有共同短板:**都改变了模型的 forward 结构,推理时必须保留 PEFT 组件**。

工业部署最看重两点:
- **质量** — 接近全参微调
- **零推理开销** — 部署后跟全参微调一样快

Hu 等人(Microsoft,2021 年 6 月,ICLR 2022)的 LoRA 论文给出了关键洞察:**对一个预训练好的权重 W₀,任务微调时的更新 ΔW 实际是"低秩"的——大部分 task-specific 信息可以用 r ≪ d 的子空间表达**。

具体公式:把 ΔW 分解为 $B \cdot A$,$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$,$r$ 通常是 8-32。训练时只更新 B, A;推理时把 BA 合并回 W₀ → **零推理延迟**。

LoRA 发布后 6 个月,从论文 demo 变成 Hugging Face PEFT 库的旗舰功能。2023 年 LLaMA 开源后,LoRA 成为开源 LLM 微调的事实标准——LoRA + LLaMA 是 2023 年 LLM 圈最常见的组合。**今天 90% 的开源 LLM 微调都基于 LoRA**。

## 核心思想:Low-Rank Decomposition of ΔW

### 假设

对一个全参微调,权重更新是:

$$
W = W_0 + \Delta W
$$

LoRA 的核心假设:**$\Delta W$ 的内在秩很低**,可以用低秩矩阵分解:

$$
\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d
$$

为什么 ΔW 是低秩的?直觉是 pretrain 已经学到了大部分通用表示,task-specific 调整只需在一个小子空间里。论文的实验也验证了——r=1 都能 work(质量微跌),r=4-8 接近 r=64。

### LoRA 层结构

对 Transformer 每个被适配的权重矩阵 $W \in \mathbb{R}^{d \times d}$(通常是 Q、V、K、O,或 FFN 的 W_in、W_out):

```
x → W₀ · x  (frozen base, no grad)
   ↘
     B · (A · x)  (trainable LoRA)
   ↗
  + → output
```

数学上:

$$
h = W_0 x + B A x = (W_0 + BA) x
$$

训练时 W₀ 冻结,只对 A, B 反传梯度。

### 初始化

- $A$ 用 Kaiming 初始化(正态小值)
- $B$ 初始化为 **全 0**

这样 $BA = 0$,训练开始时 LoRA 完全不影响 base 模型,h = W₀ x,从冻结模型平滑过渡到微调状态。

### 参数账

对 attention 的 Q、V(两个 $d × d$ 矩阵)用 LoRA:

- 全参参数:$2 \times d^2$(d=4096 时 ~33M)
- LoRA 参数:$2 \times (2 \times d \times r) = 4dr$(r=8 时 131K)
- **节省 250×**

LLaMA-7B 的 LoRA(默认 r=8,target Q+V):**~4.2M 参数,base 6.7B 的 0.06%**。

### α / Dropout / 推理合并

LoRA 实现里几个工程细节:

- **Scaling α**:$h = W_0 x + \frac{\alpha}{r} BA x$,调 α 可以独立控制 LoRA 影响幅度
- **Dropout**:在 A 后加 dropout,正则
- **推理合并**:训完把 $W_0 + BA$ 算出来作为新权重,LoRA 部分消失,推理零开销

合并后 LoRA 部分完全消失,部署时 = 全参微调的速度。这是 LoRA 与 Adapter / Prefix 的本质区别。

### LoRA 应用到哪些层

论文做了大量消融:

- **只 Q+V** ≈ Q+K+V+O ≈ 全部层 + FFN(微差几个点)
- **Q+V 是性价比最高的选择**(参数最少,质量最好)

后来工程实践扩展到 attention 全部 (Q,K,V,O) + FFN(W_up,W_down,W_gate),叫 **all-target LoRA**,但 base + ~1% 参数。

## 关键代码

最小 LoRA 层实现(PyTorch):

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Linear layer 加 LoRA 适配."""
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features

        # A: (r, in_dim), B: (out_dim, r)
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B 保持 0 初始化

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.A.T @ self.B.T
        return base_out + lora_out * self.scaling

    def merge(self):
        """训练完后合并 LoRA 到 base 权重,推理时零开销."""
        with torch.no_grad():
            self.base.weight.data += self.scaling * (self.B @ self.A)


# 给 LLaMA 的 attention 加 LoRA
def add_lora_to_llama(model, r=8, alpha=16):
    for layer in model.model.layers:
        # 默认只 Q + V
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r, alpha)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r, alpha)

# 用 HuggingFace peft 库,一行调用
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

训练 + 保存 + 加载 + 合并:

```python
# 训练
trainer.train()
# 保存 LoRA(只几 MB)
model.save_pretrained("./my-lora-adapter")

# 加载
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base, "./my-lora-adapter")

# 合并并保存为完整模型
merged = model.merge_and_unload()
merged.save_pretrained("./my-finetuned-llama")
```

## 性能数据

Hu 2021 原论文在 GPT-3 175B 上对比:

| Method | 训练参数 | WikiSQL Acc | MNLI-m Acc | SAMSum ROUGE-L |
|------|------|------|------|------|
| GPT-3 Full FT | 175B | 73.8 | 89.5 | 44.5 |
| BitFit | 14M | 71.3 | 91.0 | 40.6 |
| Prefix Tuning | 35M | 71.7 | 89.3 | 42.6 |
| Adapter(H) | 41M | 72.3 | 91.5 | 42.4 |
| **LoRA(r=4)** | **4.7M** | **73.4** | **91.7** | **44.0** |
| **LoRA(r=8)** | **9.4M** | **74.0** | **91.6** | **44.3** |

关键观察:

- **LoRA 用 GPT-3 全参的 0.005% 参数达到 95%+ 性能**
- **LoRA 比 Adapter 同质量下参数少 5-10×**
- **r=4 和 r=8 几乎没差** —— ΔW 确实低秩

LLaMA-2-7B 上现代 LoRA 实测(社区数据,r=16,target Q+V+K+O):

| Method | 训练参数 | MMLU | GSM8K | Train time(A100×1) |
|------|------|------|------|------|
| Full FT | 6.7B | 47.5 | 27.0 | OOM(单卡) |
| **LoRA r=16** | 4.5M | **47.1** | **26.5** | **4 小时** |

LoRA 几乎不掉点,但单 A100 24 小时全参 OOM,LoRA 4 小时跑完。这是 LoRA 在开源社区起飞的根本原因。

## 影响 / 后续

LoRA 在 LLM 历史的位置:**PEFT 工业标准,开源 LLM 时代微调的事实方法**。

**1. 开源 LLM 微调标配** —— 2023 年 LLaMA 开源后,Alpaca / Vicuna / WizardLM / Code LLaMA 等几乎所有开源微调模型都用 LoRA。Hugging Face PEFT 库 GitHub star 在 LLaMA 发布后 6 个月内从 5K 涨到 15K+

**2. LoRA Hub / civitai 生态** —— 类似 Stable Diffusion 的 LoRA 概念,LLM 的 LoRA 也催生了生态。HuggingFace 上有数十万个 LoRA adapter(每个几 MB),用户可以下载 + 即插即用

**3. [QLoRA](04-qlora.md)(2023.5)的基础** —— Dettmers 在 LoRA 之上加 4-bit 量化,把 65B 模型微调拉到单卡 24GB 消费级 GPU。QLoRA 是 LoRA 的直接扩展

**4. LoRA 变体爆发** —— DoRA(2024)、LoRA+(2024)、AdaLoRA、LoRA-FA、VeRA、PiSSA 等几十种变体,每个尝试改进 LoRA 的某个方面(收敛速度、最终质量、合并方式)。但多数没显著超越原 LoRA

**5. 多 LoRA 服务(LoRA serving)** —— S-LoRA / LoRAX 等系统让单 GPU 同时 serve 数千个 LoRA(用户级 / 任务级 LoRA),开启 "LoRA-as-a-Service" 新场景

**6. LoRA 在 RLHF / DPO 中普遍应用** —— [InstructGPT](../12-rlhf-alignment/02-instructgpt.md) / DPO 训练时用 LoRA 大幅降低 RL 阶段成本。OpenAI / Anthropic 内部据传也用 LoRA 做某些 alignment 任务

**7. LoRA 在 multimodal / diffusion 模型上的扩展** —— Stable Diffusion 的 LoRA(Concept LoRA)是 SD 生态最重要的扩展机制。一个 base SD + 一个角色 LoRA + 一个画风 LoRA 是社区标准用法

LoRA 留下的开放问题:

- **rank r 怎么自适应** —— 不同任务 / 不同层最优 r 不同 → AdaLoRA / SaLoRA
- **多 LoRA 组合的干扰** —— 多个 LoRA 同时加载时 BA 相加可能互相干扰 → DARE / TIES Merging
- **MoE + LoRA** —— MoE 模型的 LoRA 怎么放(每 expert 一个?共享 router?)→ MoLE 等工作
- **LoRA + DPO 的稳定性** —— 大学习率 + 小参数 在 RLHF 时偶发不稳定 → LoRA+ / ORPO

→ [04-qlora.md](04-qlora.md) · LoRA + 4-bit 量化,消费级硬件微调 65B
→ [01-adapter.md](01-adapter.md) · LoRA 的父思想,可视为线性化的 adapter
→ [02-prefix-tuning.md](02-prefix-tuning.md) · 同期 PEFT 方法,被 LoRA 工程上替代
→ [../12-rlhf-alignment/04-dpo.md](../12-rlhf-alignment/04-dpo.md) · DPO 训练几乎都用 LoRA
→ [../15-reasoning-o1-r1/04-deepseek-r1.md](../15-reasoning-o1-r1/04-deepseek-r1.md) · R1 蒸馏模型的进一步微调普遍用 LoRA
→ [../13-moe-efficient/03-mixtral.md](../13-moe-efficient/03-mixtral.md) · Mixtral 的 LoRA 微调是社区主要使用方式
