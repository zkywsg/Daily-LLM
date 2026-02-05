# 模型微调PEFT

[English](README.md) | [中文](README_CN.md)

## 目录

1. [背景](#1-背景)
2. [核心概念](#2-核心概念)
3. [数学原理](#3-数学原理)
4. [代码实现](#4-代码实现)
5. [实验对比](#5-实验对比)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景

### 1.1 全量微调的问题

- **显存需求**: 需要存储完整梯度
- **训练时间**: 参数多训练慢
- **灾难遗忘**: 容易忘记预训练知识
- **存储成本**: 每个任务一个完整模型

### 1.2 PEFT优势

- **参数高效**: 只训练少量参数
- **显存节省**: 70%+显存节省
- **快速训练**: 训练速度提升3-5x
- **避免遗忘**: 保留预训练知识

---

## 2. 核心概念

### 2.1 LoRA (Low-Rank Adaptation)

**核心思想**: 用低秩矩阵近似权重更新

**公式**: $W = W_0 + \Delta W = W_0 + BA$

- $W_0$: 预训练权重 (冻结)
- $B, A$: 可训练的低秩矩阵

### 2.2 QLoRA

LoRA + 4-bit量化，进一步节省显存:
- 65B模型可在48GB显存训练

### 2.3 Adapter

在Transformer层插入小型适配器模块。

---

## 3. 数学原理

### 3.1 低秩分解

$$
\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
$$

其中 $r \ll \min(d, k)$ 是低秩维度。

### 3.2 参数量对比

**全量微调**:
$$N_{\text{full}} = d \times k$$

**LoRA**:
$$N_{\text{lora}} = d \times r + r \times k = r(d + k)$$

**节省比例**:
$$\frac{N_{\text{lora}}}{N_{\text{full}}} = \frac{r(d+k)}{dk} \approx \frac{2r}{k} \quad (\text{当 } d \approx k)$$

---

## 4. 代码实现

### 4.1 LoRA微调

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # 低秩维度
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 33M || all params: 7B || trainable%: 0.47

# 训练
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        output_dir="./lora_model"
    )
)

trainer.train()

# 保存LoRA权重
model.save_pretrained("./lora_weights")

# 推理时合并权重 (可选)
# model = model.merge_and_unload()
```

### 4.2 QLoRA微调

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备模型用于训练
model = prepare_model_for_kbit_training(model)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 训练 (与LoRA相同)
trainer = SFTTrainer(model=model, ...)
trainer.train()
```

---

## 5. 实验对比

### 5.1 性能对比

| 方法 | 可训练参数 | 显存需求 | 训练时间 | 效果 |
|------|-----------|---------|---------|------|
| **全量微调** | 100% | 100% | 100% | 100% |
| **LoRA (r=16)** | 0.5% | 30% | 25% | 98% |
| **QLoRA** | 0.5% | 18% | 30% | 97% |
| **Adapter** | 1% | 35% | 30% | 96% |

### 5.2 不同rank效果

| Rank | 参数占比 | 下游任务效果 |
|------|---------|-------------|
| 4 | 0.12% | 92% |
| 8 | 0.24% | 95% |
| 16 | 0.47% | 98% |
| 32 | 0.94% | 99% |
| 64 | 1.88% | 99.5% |

**推荐**: r=16是性价比最佳点

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **选择合适rank**: 一般r=8-32
2. **目标模块**: 优先q_proj, v_proj
3. **学习率**: LoRA通常需要更高学习率 (1e-4 to 2e-4)
4. **alpha**: 通常设置为2*rank
5. **数据质量**: PEFT对数据质量更敏感

### 6.2 常见陷阱

1. **rank过低**: 表达能力不足
2. **模块选择不当**: 不是所有层都需要LoRA
3. **学习率过小**: 训练不充分
4. **数据不平衡**: 某些类别过拟合

### 6.3 模块选择建议

```markdown
推荐目标模块:
- ✓ q_proj, v_proj (必须)
- ✓ k_proj, o_proj (推荐)
- ? gate_proj, up_proj, down_proj (MLP层，可选)
- ✗ embed_tokens, lm_head (通常不需要)
```

---

## 7. 总结

PEFT让大模型微调变得高效可行：

1. **LoRA**: 最常用，效果与效率平衡
2. **QLoRA**: 显存极度受限时使用
3. **Adapter**: 多任务场景
4. **Prompt Tuning**: 极轻量级

**选择建议**:
- 单卡/有限显存 → QLoRA
- 追求最佳效果 → LoRA (r=16-32)
- 多任务场景 → Adapter
- 快速实验 → Prompt Tuning

**关键参数**:
- r=16, alpha=32
- 目标模块: q_proj, v_proj
- 学习率: 2e-4
- dropout: 0.05
