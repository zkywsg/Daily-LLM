# 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)

[English](README_EN.md) | [中文](README.md)

## 概述

PEFT 方法通过仅训练少量参数来调整大型预训练模型以适应下游任务,大幅降低内存需求和训练成本,同时保持竞争力性能。

## 为什么需要 PEFT?

### 全量微调的挑战

| Challenge | Impact |
|-----------|--------|
| **内存** | 存储所有参数的梯度 |
| **存储** | 每个任务需要完整的模型副本 |
| **灾难性遗忘** | 丢失预训练知识 |
| **计算** | 大模型训练时间长 |

**示例: LLaMA-2 70B 全量微调**
- 模型参数: 140GB (FP16)
- 梯度: 140GB
- 优化器状态 (Adam): 280GB
- **总计: ~560GB GPU 内存**

### PEFT 的优势

| Benefit | Improvement |
|---------|-------------|
| **内存** | 减少 70-90% |
| **存储** | 仅保存小型适配器权重 |
| **训练速度** | 快 3-5 倍 |
| **知识保留** | 基础模型冻结 |

## LoRA (Low-Rank Adaptation, 低秩适应)

### 核心概念

不是更新完整的权重矩阵 $W \in \mathbb{R}^{d \times k}$,LoRA 将更新分解为低秩矩阵:

```
W = W_0 + ΔW = W_0 + BA

其中:
- W_0: 冻结的预训练权重 (d × k)
- B: 可训练矩阵 (d × r)
- A: 可训练矩阵 (r × k)
- r: 低秩维度 (r ≪ min(d, k))
```

**前向传播**:
```
h = W_0·x + ΔW·x = W_0·x + B·A·x
```

### 数学分析

**参数数量**:

| Method | Formula | Example (d=k=4096, r=16) |
|--------|---------|------------------------|
| 全量微调 | d × k | 16.8M |
| LoRA | r × (d + k) | 131K |
| **节省** | - | **99.2%** |

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, lora_alpha=32):
        super().__init__()

        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # LoRA 矩阵 - 使用特定策略初始化
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # A 用 kaiming uniform 初始化,B 用零初始化
        # 这确保 ΔW 在初始化时为零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, base_output):
        """
        x: 输入张量
        base_output: 来自冻结基础层的输出
        """
        # LoRA 路径: x @ A @ B
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling

        return base_output + lora_output

# 完整的 LoRA 线性层
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, lora_alpha=32, dropout=0.0):
        super().__init__()

        self.base_layer = base_layer
        # 冻结基础层
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 添加 LoRA
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora = LoRALayer(in_features, out_features, rank, lora_alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 基础输出 (冻结)
        base_output = self.base_layer(x)

        # LoRA 适应
        x_dropped = self.dropout(x)
        return self.lora(x_dropped, base_output)
```

### LoRA 配置

```python
from peft import LoraConfig, get_peft_model, TaskType

# 标准 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # 低秩维度
    lora_alpha=32,          # 缩放因子 (通常 2*r)
    target_modules=[        # 要适应的层
        "q_proj",           # Query 投影
        "k_proj",           # Key 投影
        "v_proj",           # Value 投影
        "o_proj",           # 输出投影
    ],
    lora_dropout=0.05,      # Dropout 用于正则化
    bias="none",            # 偏置训练策略
    modules_to_save=None,   # 额外要训练的模块
)

# 应用于模型
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 转换为 PEFT 模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# 输出: trainable params: 33,554,432 || all params: 6,771,970,048 ||
#         trainable%: 0.4956
```

### 目标模块选择

| Module | Recommendation | Reason |
|--------|---------------|--------|
| **q_proj, v_proj** | ✓ 必需 | 对注意力最重要 |
| **k_proj, o_proj** | ✓ 推荐 | 提高性能 |
| **gate_proj, up_proj, down_proj** | ? 可选 | MLP 层,更多参数 |
| **embed_tokens** | ✗ 通常跳过 | 大词表维度 |
| **lm_head** | ✗ 通常跳过 | 输出层 |

## QLoRA (Quantized LoRA, 量化 LoRA)

### 4-bit 量化 + LoRA

QLoRA 通过以下方式在单张 48GB GPU 上微调 65B 模型:
1. 将基础模型量化为 4-bit
2. 在更高精度下计算 LoRA 更新
3. 在前向传播期间即时去量化

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 使用 4-bit 量化
    bnb_4bit_use_double_quant=True,             # 嵌套量化
    bnb_4bit_quant_type="nf4",                  # 4-bit normal float
    bnb_4bit_compute_dtype=torch.bfloat16       # 计算数据类型
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                          # 自动分发层
    trust_remote_code=True
)

# 准备训练
model = prepare_model_for_kbit_training(model)

# 应用 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 内存对比
print("QLoRA enables training 65B models on 48GB GPU")
print("Standard LoRA requires ~80GB for 65B model")
print("Full fine-tuning requires ~780GB for 65B model")
```

### 内存节省分解

| Model Size | Full FT | LoRA | QLoRA |
|------------|---------|------|-------|
| 7B | 28GB | 14GB | 8GB |
| 13B | 52GB | 26GB | 14GB |
| 70B | 280GB | 140GB | 48GB |

## Adapters (适配器)

### 架构

在 transformer 层之间插入的小型瓶颈模块:

```
Input → [LayerNorm → Adapter → Residual] → Output

Adapter 结构:
x → Linear(down) → Activation → Linear(up) → Output
```

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64, dropout=0.1):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.xavier_uniform_(self.up_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return x + residual  # 残差连接

# Adapter 配置
from peft import AdapterConfig

adapter_config = AdapterConfig(
    adapter_dim=64,          # 瓶颈维度
    hidden_act="gelu",       # 激活函数
    adapter_dropout=0.1,
    target_modules=["attention", "mlp"]
)
```

## Prompt Tuning (提示词微调)

### Soft Prompts (软提示词)

学习连续嵌入而非离散文本提示词:

```python
class PromptTuning(nn.Module):
    def __init__(self, num_tokens, token_dim, initialize_from_vocab=True):
        super().__init__()

        # 初始化软提示词 tokens
        if initialize_from_vocab:
            # 从现有词表初始化
            self.soft_prompt = nn.Parameter(
                torch.randn(num_tokens, token_dim) * 0.01
            )
        else:
            # 随机初始化
            self.soft_prompt = nn.Parameter(
                torch.randn(num_tokens, token_dim)
            )

    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)

        # 扩展软提示词以适应 batch
        soft_prompt_embeds = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # 拼接: [soft prompt] + [input]
        return torch.cat([soft_prompt_embeds, input_embeds], dim=1)

# 与 transformer 一起使用
class PromptTunedTransformer(nn.Module):
    def __init__(self, base_model, num_prompt_tokens=20):
        super().__init__()
        self.base_model = base_model

        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 可学习的软提示词
        self.prompt_tuning = PromptTuning(
            num_tokens=num_prompt_tokens,
            token_dim=self.base_model.config.hidden_size
        )

    def forward(self, input_ids):
        # 获取输入嵌入
        input_embeds = self.base_model.embeddings(input_ids)

        # 添加软提示词
        prompted_embeds = self.prompt_tuning(input_embeds)

        # 通过冻结的 transformer
        return self.base_model(inputs_embeds=prompted_embeds)
```

## P-tuning v2

### 深度提示词微调

在每一层添加可训练提示词,不仅仅是输入:

```python
class PTuningV2(nn.Module):
    def __init__(self, num_layers, num_tokens, hidden_size):
        super().__init__()

        # 每层的提示词嵌入
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.01)
            for _ in range(num_layers)
        ])

        # 提示词生成的 MLP (可选)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def forward(self, hidden_states, layer_idx):
        """在特定层将提示词添加到隐藏状态"""
        batch_size = hidden_states.size(0)

        # 获取该层的提示词
        prompts = self.prompt_embeddings[layer_idx]
        prompts = self.prompt_encoder(prompts)

        # 扩展以适应 batch
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)

        # 拼接
        return torch.cat([prompts, hidden_states], dim=1)
```

## 对比

| Method | Trainable Params | Memory | Speed | Performance |
|--------|-----------------|--------|-------|-------------|
| **Full FT** | 100% | 100% | 1x | 100% |
| **LoRA** | 0.5-2% | 30-40% | 4x | 98% |
| **QLoRA** | 0.5-2% | 20-25% | 3x | 97% |
| **Adapter** | 1-3% | 35-45% | 3x | 96% |
| **Prompt Tuning** | <0.1% | 15-20% | 5x | 90% |
| **P-tuning v2** | 0.2-0.5% | 20-25% | 4x | 94% |

## 超参数调优

### LoRA 秩选择

| Rank | Parameters | Performance | Use Case |
|------|-----------|-------------|----------|
| 4 | 0.12% | 92% | 快速实验 |
| 8 | 0.24% | 95% | 资源受限 |
| 16 | 0.47% | 98% | **推荐** |
| 32 | 0.94% | 99% | 高性能 |
| 64 | 1.88% | 99.5% | 最大质量 |

### 学习率指南

| Model Size | LoRA LR | Full FT LR |
|------------|---------|------------|
| < 1B | 1e-3 | 5e-5 |
| 7B | 1e-4 | 2e-5 |
| 13B | 1e-4 | 1e-5 |
| 70B+ | 5e-5 | 5e-6 |

## 最佳实践

### 1. 模块选择

```python
# 最小配置 (最快)
target_modules = ["q_proj", "v_proj"]

# 推荐 (最佳平衡)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 全面 (最高质量)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### 2. 训练配置

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,           # 比全量微调高
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",     # 用于 QLoRA
    group_by_length=True,        # 效率
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()
```

### 3. 推理和部署

```python
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载 LoRA 权重
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 选项 1: 合并权重 (推理更快,无需 PEFT 依赖)
model = model.merge_and_unload()
model.save_pretrained("./merged_model")

# 选项 2: 保持分离 (内存高效,易于交换适配器)
# 直接使用模型进行推理
```

## 常见陷阱

| Issue | Symptom | Solution |
|-------|---------|----------|
| **秩太低** | 性能差 | 将 r 增加到 16+ |
| **错误的模块** | 无改进 | 至少使用 q_proj, v_proj |
| **学习率太低** | 收敛慢 | 使用全量微调的 10-100x LR |
| **过拟合** | 训练损失高,验证损失低 | 添加 dropout,减少 epochs |
| **不稳定** | 损失尖峰 | 梯度裁剪,warmup |

---

**上一节**: [预训练](../pre-training/README.md) | **下一节**: [对齐](../alignment/README.md)
