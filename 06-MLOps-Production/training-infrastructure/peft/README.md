# PEFT Fine-tuning

**[English](README.md) | [中文](README_CN.md)**

## Table of Contents

1. [Background](#1-background)
2. [Core Concepts](#2-core-concepts)
3. [Mathematical Principles](#3-mathematical-principles)
4. [Code Implementation](#4-code-implementation)
5. [Experimental Comparison](#5-experimental-comparison)
6. [Best Practices and Common Pitfalls](#6-best-practices-and-common-pitfalls)
7. [Summary](#7-summary)

---

## 1. Background

### 1.1 Problems with Full Fine-tuning

- **Memory Requirements**: Need to store complete gradients
- **Training Time**: Many parameters, slow training
- **Catastrophic Forgetting**: Easy to forget pre-trained knowledge
- **Storage Cost**: One complete model per task

### 1.2 PEFT Advantages

- **Parameter Efficient**: Only train small number of parameters
- **Memory Savings**: 70%+ memory savings
- **Fast Training**: 3-5x training speed improvement
- **Avoid Forgetting**: Preserve pre-trained knowledge

---

## 2. Core Concepts

### 2.1 LoRA (Low-Rank Adaptation)

**Core Idea**: Approximate weight updates with low-rank matrices

**Formula**: $W = W_0 + \Delta W = W_0 + BA$

- $W_0$: Pre-trained weights (frozen)
- $B, A$: Trainable low-rank matrices

### 2.2 QLoRA

LoRA + 4-bit quantization, further saves memory:
- 65B models can be trained on 48GB GPU memory

### 2.3 Adapter

Insert small adapter modules in Transformer layers.

---

## 3. Mathematical Principles

### 3.1 Low-Rank Decomposition

$$
\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
$$

Where $r \ll \min(d, k)$ is the low-rank dimension.

### 3.2 Parameter Count Comparison

**Full Fine-tuning**:
$$N_{\text{full}} = d \times k$$

**LoRA**:
$$N_{\text{lora}} = d \times r + r \times k = r(d + k)$$

**Savings Ratio**:
$$\frac{N_{\text{lora}}}{N_{\text{full}}} = \frac{r(d+k)}{dk} \approx \frac{2r}{k} \quad (\text{when } d \approx k)$$

---

## 4. Code Implementation

### 4.1 LoRA Fine-tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Low-rank dimension
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 33M || all params: 7B || trainable%: 0.47

# Training
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

# Save LoRA weights
model.save_pretrained("./lora_weights")

# Merge weights at inference (optional)
# model = model.merge_and_unload()
```

### 4.2 QLoRA Fine-tuning

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training (same as LoRA)
trainer = SFTTrainer(model=model, ...)
trainer.train()
```

---

## 5. Experimental Comparison

### 5.1 Performance Comparison

| Method | Trainable Params | Memory Requirement | Training Time | Effect |
|--------|-----------------|-------------------|----------------|--------|
| **Full Fine-tuning** | 100% | 100% | 100% | 100% |
| **LoRA (r=16)** | 0.5% | 30% | 25% | 98% |
| **QLoRA** | 0.5% | 18% | 30% | 97% |
| **Adapter** | 1% | 35% | 30% | 96% |

### 5.2 Different Rank Effects

| Rank | Param Ratio | Downstream Task Effect |
|------|-------------|---------------------|
| 4 | 0.12% | 92% |
| 8 | 0.24% | 95% |
| 16 | 0.47% | 98% |
| 32 | 0.94% | 99% |
| 64 | 1.88% | 99.5% |

**Recommended**: r=16 is the best cost-effectiveness point

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

1. **Choose appropriate rank**: Generally r=8-32
2. **Target modules**: Prioritize q_proj, v_proj
3. **Learning rate**: LoRA typically needs higher learning rate (1e-4 to 2e-4)
4. **Alpha**: Usually set to 2*rank
5. **Data quality**: PEFT is more sensitive to data quality

### 6.2 Common Pitfalls

1. **Rank too low**: Insufficient expressive power
2. **Improper module selection**: Not all layers need LoRA
3. **Learning rate too small**: Insufficient training
4. **Data imbalance**: Some classes overfitting

### 6.3 Module Selection Recommendations

```markdown
Recommended target modules:
- ✓ q_proj, v_proj (required)
- ✓ k_proj, o_proj (recommended)
- ? gate_proj, up_proj, down_proj (MLP layers, optional)
- ✗ embed_tokens, lm_head (usually not needed)
```

---

## 7. Summary

PEFT makes large model fine-tuning efficient and feasible:

1. **LoRA**: Most commonly used, balances effect and efficiency
2. **QLoRA**: Use when GPU memory is extremely limited
3. **Adapter**: Multi-task scenarios
4. **Prompt Tuning**: Ultra-lightweight

**Selection Guidelines**:
- Single GPU / limited memory → QLoRA
- Pursue best effect → LoRA (r=16-32)
- Multi-task scenarios → Adapter
- Quick experiments → Prompt Tuning

**Key Parameters**:
- r=16, alpha=32
- Target modules: q_proj, v_proj
- Learning rate: 2e-4
- Dropout: 0.05
