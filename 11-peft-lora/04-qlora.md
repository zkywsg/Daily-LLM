---
name: "QLoRA"
year: 2023
family: "11-peft-lora"
order: 4
paper: "QLoRA: Efficient Finetuning of Quantized LLMs"
authors: ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"]
key_idea: "Base 模型量化到 4-bit NF4(NormalFloat)+ LoRA 微调,配 double quantization + paged optimizer;让 65B 模型能在单卡 48GB(实际 24GB 也能)GPU 上微调,LLM 微调彻底个人化"
---

## 前作进展

2023 年中,LoRA 已成开源 LLM 微调标配,但仍有一个根本限制:**base 模型必须以 fp16/bf16 加载**。

具体显存账(LLaMA 系列,fp16):

| 模型 | base 加载 | LoRA 训练优化器状态 | 总显存 |
|------|------|------|------|
| LLaMA-7B | 14 GB | ~4 GB | ~20 GB |
| LLaMA-13B | 26 GB | ~6 GB | ~35 GB |
| LLaMA-33B | 66 GB | ~10 GB | ~80 GB |
| LLaMA-65B | 130 GB | ~15 GB | ~150 GB |

对照消费级硬件:

- RTX 3090(24GB):勉强 7B
- RTX 4090(24GB):勉强 13B
- A100 40GB:勉强 33B
- A100 80GB:勉强 65B 但 batch 极小

65B 模型微调需要 2× A100-80GB,**单 GPU 没法跑 LLaMA-65B**。社区拿到了开源 65B 权重但没法定制——这就像有了汽车但没有汽油。

Dettmers 等人(华盛顿大学,2023 年 5 月)的 QLoRA 论文给出关键突破:**把 base 模型量化到 4-bit,LoRA 部分保持 fp16,梯度只反传到 LoRA**。这样 65B 模型加载从 130GB 降到 ~33GB,**单卡 48GB(如 A6000)就能微调 65B**。论文还提出了"使用 QLoRA 训练的 Guanaco-65B 模型,达到 ChatGPT 99.3% 性能"的轰动结果。

QLoRA 发布后:

- bitsandbytes(Dettmers 自己的量化库)star 数 1 个月翻 5×
- HuggingFace 把 QLoRA 集成进 `transformers` 库默认 quantization 接口
- 个人开发者第一次能在自己的 4090 / A6000 上微调 70B+ 模型

**QLoRA 把 LLM 微调从"机构级"工程降到"个人级"工程**。今天的开源大模型微调(LLaMA-3-70B, Qwen-2.5-72B, DeepSeek-V2-Lite)绝大多数都用 QLoRA。

## 核心思想:4-bit Base + LoRA + 系统优化

QLoRA 不是单一 trick,是几个量化 / 训练优化的组合:

### 1. NF4(NormalFloat 4-bit)量化

标准 4-bit 量化(INT4)用均匀分布,但神经网络权重是**正态分布**。Dettmers 提出 **NF4**——基于正态分布的最优 4-bit quantization:

- 把 fp16 权重的 16 个量化 level 选在正态分布的 16 个等分位点
- 比 INT4 误差小 ~30%
- 比 INT8 接近(NF4 是 4-bit,但实际精度损失小)

具体地,NF4 的 16 个值是预计算好的(标准正态的 16 等分位点):

```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1849, -0.0911, 0.0,
 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

量化时把每 block 的 fp16 权重 normalize 到 [-1, 1],然后映射到这 16 个值之一。

### 2. Double Quantization(双重量化)

NF4 量化时每 block(64 个值)需要存一个 fp32 scale factor。256 个 block 就要 256 × 32 bit = 8KB 额外开销。Double Quantization 把这些 scale factor 自身再 8-bit 量化,进一步省 ~30% 显存。

### 3. Paged Optimizer

训练时 AdamW 优化器状态(Adam 的 m, v)显存占用大。QLoRA 用 NVIDIA Unified Memory 把优化器状态"分页"——大部分留在 CPU 内存,GPU 训练时按需 paging 到 GPU。结合 4-bit base,把 65B 训练显存从 150GB 压到 ~33GB。

### 4. 训练流程

```
Step 1: 加载 base 模型,量化到 4-bit NF4
   LLaMA-65B fp16:130GB → 4-bit NF4:33GB

Step 2: 加载 LoRA adapter,fp16
   LoRA 参数 ~ 30M × 2 bytes = 60MB

Step 3: forward:
   - base 权重以 4-bit 存储,计算时反量化到 bf16 做矩阵乘
   - LoRA 始终 bf16
   - 输出 = W₀_dequant @ x + BA @ x

Step 4: backward:
   - 只更新 A, B(LoRA 参数)
   - base 完全不更新,也不需要 base 的梯度
   - 优化器状态 paged 到 CPU

Step 5: 推理:
   - 量化版 base + LoRA fp16,推理时合并(或不合并保留 LoRA 灵活性)
```

### 显存账(LLaMA-65B QLoRA)

| 组件 | fp16 LoRA | **QLoRA** |
|------|------|------|
| Base 模型 | 130 GB | **33 GB**(NF4 + DQ) |
| LoRA params | 60 MB | 60 MB |
| Optimizer state | 15 GB | **3 GB**(paged) |
| Activation | 5 GB | 5 GB |
| **总计** | **~150 GB** | **~41 GB** |

实测可以在单卡 48GB A6000 跑 LLaMA-65B QLoRA,batch=1。

## 关键代码

QLoRA 训练完整流程(HuggingFace transformers + peft + bitsandbytes):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_use_double_quant=True,      # double quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化到 bf16
)

# 2. 加载 4-bit base 模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# 3. 准备 kbit training(关键步骤,处理 layer norm 等需要 fp32 的层)
model = prepare_model_for_kbit_training(model)

# 4. 添加 LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # all-target
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~200M, all params: 70B (with 4-bit base), trainable: ~0.3%

# 5. 训练(用 paged optimizer)
from transformers import Trainer, TrainingArguments
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./qlora-llama-70b",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",  # paged optimizer
        max_steps=1000,
        save_steps=100,
        gradient_checkpointing=True,  # 进一步省显存
    ),
    train_dataset=dataset,
)
trainer.train()

# 6. 保存
model.save_pretrained("./qlora-llama-70b-adapter")  # 只 LoRA 部分(几 GB)

# 7. 推理(加载 4-bit base + LoRA)
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./qlora-llama-70b-adapter")
inputs = tokenizer("解释 QLoRA 的核心思想:", return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=200)[0]))
```

## 性能数据

Dettmers 2023 用 QLoRA 训练 Guanaco 模型系列(基于 LLaMA),与 ChatGPT / GPT-4 / Vicuna 对比:

| Model | 训练硬件 | 训练时间 | Vicuna Eval(vs GPT-4 评分) |
|------|------|------|------|
| ChatGPT(2023.3) | - | - | **100%** |
| GPT-4 | - | - | 119% |
| Vicuna-13B(FP16 LoRA) | 8× A100-40GB | ~1 day | 92% |
| **Guanaco-7B(QLoRA)** | **单 RTX 3090** | **12 小时** | **87%** |
| **Guanaco-13B(QLoRA)** | **单 A100-40GB** | **12 小时** | **93%** |
| **Guanaco-33B(QLoRA)** | **单 A6000-48GB** | **18 小时** | **97%** |
| **Guanaco-65B(QLoRA)** | **单 A6000-48GB** | **24 小时** | **99.3%** |

关键观察:

- **Guanaco-65B 99.3% ChatGPT 性能** —— 用单卡 48GB GPU 训出来的模型与商业级 ChatGPT 几乎持平
- **训练成本** —— A6000 时租 ~$1.5/h × 24 = **$36**。对比 OpenAI 训 GPT-3.5 估 $5M
- **个人能跑 65B 微调** —— A6000 是高端但消费可得(~$5000),已远低于机构级 8× A100 集群

QLoRA 与 FP16 LoRA 对比(LLaMA-7B,Alpaca 微调):

| Method | 训练显存 | MMLU | TruthfulQA |
|------|------|------|------|
| Full FT(fp16) | 80 GB | 47.0 | 26.5 |
| LoRA(fp16) | 20 GB | 46.8 | 26.7 |
| **QLoRA(NF4)** | **5 GB** | **46.5** | **26.5** |

QLoRA 显存压缩 16×,质量几乎不掉。

## 影响 / 后续

QLoRA 在 LLM 历史的位置:**LLM 微调民主化转折点,让千万个人开发者拿到生产级 LLM 定制能力**。

**1. 开源 LLM 微调生态彻底改变** —— QLoRA 发布后,几乎所有开源 LLM 微调教程 / Colab 默认用 QLoRA。LLaMA-2 / LLaMA-3 / Mistral / Qwen / DeepSeek 等开源模型的微调几乎 100% 走 QLoRA 路线

**2. NF4 / Double Quantization 成 4-bit 量化标准** —— bitsandbytes 库的 NF4 实现成为 HuggingFace 默认 4-bit 加载选项。后续 AWQ / GPTQ 等量化方法被用于推理,QLoRA NF4 主要用于训练

**3. Guanaco 系列的影响** —— Guanaco-65B 作为 QLoRA 论文的 demo,证明"小团队也能训 SOTA 模型"。这一精神催生了大量小团队 / 个人微调工作

**4. Paged Optimizer 被广泛复用** —— Adam 优化器状态 paging 到 CPU 的技巧后来被 Megatron / DeepSpeed / FSDP 等系统也吸收

**5. 推动消费级 GPU 需求** —— QLoRA 让 RTX 3090/4090/A6000 在 LLM 微调上变得"够用"。一定程度上挑战了"必须 H100 集群"的认知,推动二手 GPU 市场和 cloud GPU 服务

**6. 加速 LLM 应用层创新** —— 微调成本从 $10K+ 降到 $50,催生大量垂直领域 LLM:法律 LLM、医疗 LLM、客服 LLM、代码 LLM 等。**没有 QLoRA 就没有 2024-2025 年的 LLM 应用层繁荣**

**7. 后续量化 + PEFT 工作** —— GPTQ + LoRA(QA-LoRA)、AWQ + LoRA、LoftQ(在量化时让 LoRA 初始化更好)等延续 QLoRA 思路。但 NF4 + DQ + paged optimizer 这套组合至今没被显著超越

QLoRA 留下的开放问题:

- **4-bit 是否够** —— NF4 已经是当前最优 4-bit,要不要 3-bit / 2-bit?2-bit 质量损失明显
- **训练精度问题** —— 4-bit base + bf16 LoRA 在某些 task 仍有 micro 精度损失
- **MoE + QLoRA** —— [DeepSeek-V3](../13-moe-efficient/04-deepseek-v3.md) 这类 600B+ MoE 怎么 QLoRA 还在探索
- **长上下文训练** —— 8K+ 上下文 + QLoRA 的 activation memory 仍是问题 → Liger Kernel / Unsloth 等工程优化

→ [03-lora.md](03-lora.md) · 父技术,QLoRA = LoRA + 量化
→ [01-adapter.md](01-adapter.md) · PEFT 起源
→ [02-prefix-tuning.md](02-prefix-tuning.md) · soft prompt 路线,与 QLoRA 互补
→ [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) · QLoRA 的主战场是开源 LLaMA / Mistral 系列
→ [../12-rlhf-alignment/04-dpo.md](../12-rlhf-alignment/04-dpo.md) · DPO + QLoRA 是开源 RLHF 标配
→ [../13-moe-efficient/03-mixtral.md](../13-moe-efficient/03-mixtral.md) · Mixtral 的 QLoRA 微调是社区主要使用方式
