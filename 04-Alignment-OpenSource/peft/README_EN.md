# Parameter-Efficient Fine-Tuning (PEFT)

**[English](README_EN.md) | [中文](README.md)**

## Overview

PEFT methods adapt large pre-trained models to downstream tasks by training only a small number of parameters, dramatically reducing memory requirements and training costs while maintaining competitive performance.

## Why PEFT?

### Full Fine-tuning Challenges

| Challenge | Impact |
|-----------|--------|
| **Memory** | Store gradients for all parameters |
| **Storage** | Each task requires a full model copy |
| **Catastrophic Forgetting** | Lose pre-trained knowledge |
| **Compute** | Long training times for large models |

**Example: LLaMA-2 70B Full Fine-tuning**
- Model parameters: 140GB (FP16)
- Gradients: 140GB
- Optimizer states (Adam): 280GB
- **Total: ~560GB GPU memory**

### PEFT Benefits

| Benefit | Improvement |
|---------|-------------|
| **Memory** | 70-90% reduction |
| **Storage** | Only save small adapter weights |
| **Training Speed** | 3-5x faster |
| **Knowledge Retention** | Base model frozen |

## LoRA (Low-Rank Adaptation)

### Core Concept

Instead of updating full weight matrices $W \in \mathbb{R}^{d \times k}$, LoRA decomposes the update into low-rank matrices:

```
W = W_0 + ΔW = W_0 + BA

Where:
- W_0: Frozen pre-trained weights (d × k)
- B: Trainable matrix (d × r)
- A: Trainable matrix (r × k)
- r: Low-rank dimension (r ≪ min(d, k))
```

**Forward Pass**:
```
h = W_0·x + ΔW·x = W_0·x + B·A·x
```

### Mathematical Analysis

**Parameter Count**:

| Method | Formula | Example (d=k=4096, r=16) |
|--------|---------|------------------------|
| Full Fine-tuning | d × k | 16.8M |
| LoRA | r × (d + k) | 131K |
| **Savings** | - | **99.2%** |

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
        
        # LoRA matrices - initialized with specific strategy
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with kaiming uniform, B with zeros
        # This ensures ΔW is zero at initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, base_output):
        """
        x: input tensor
        base_output: output from frozen base layer
        """
        # LoRA path: x @ A @ B
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return base_output + lora_output

# Complete LoRA Linear layer
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, lora_alpha=32, dropout=0.0):
        super().__init__()
        
        self.base_layer = base_layer
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Add LoRA
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora = LoRALayer(in_features, out_features, rank, lora_alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        # Base output (frozen)
        base_output = self.base_layer(x)
        
        # LoRA adaptation
        x_dropped = self.dropout(x)
        return self.lora(x_dropped, base_output)
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model, TaskType

# Standard LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Low-rank dimension
    lora_alpha=32,          # Scaling factor (usually 2*r)
    target_modules=[        # Which layers to adapt
        "q_proj",           # Query projection
        "k_proj",           # Key projection  
        "v_proj",           # Value projection
        "o_proj",           # Output projection
    ],
    lora_dropout=0.05,      # Dropout for regularization
    bias="none",            # Bias training strategy
    modules_to_save=None,   # Additional modules to train
)

# Apply to model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Convert to PEFT model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 33,554,432 || all params: 6,771,970,048 || 
#         trainable%: 0.4956
```

### Target Module Selection

| Module | Recommendation | Reason |
|--------|---------------|--------|
| **q_proj, v_proj** | ✓ Required | Most important for attention |
| **k_proj, o_proj** | ✓ Recommended | Improves performance |
| **gate_proj, up_proj, down_proj** | ? Optional | MLP layers, more params |
| **embed_tokens** | ✗ Usually skip | Large vocab dimension |
| **lm_head** | ✗ Usually skip | Output layer |

## QLoRA (Quantized LoRA)

### 4-bit Quantization + LoRA

QLoRA enables fine-tuning 65B models on single 48GB GPU by:
1. Quantizing base model to 4-bit
2. Computing LoRA updates in higher precision
3. Dequantizing on-the-fly during forward pass

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Use 4-bit quantization
    bnb_4bit_use_double_quant=True,             # Nested quantization
    bnb_4bit_quant_type="nf4",                  # 4-bit normal float
    bnb_4bit_compute_dtype=torch.bfloat16       # Compute dtype
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                          # Auto-distribute layers
    trust_remote_code=True
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Memory comparison
print("QLoRA enables training 65B models on 48GB GPU")
print("Standard LoRA requires ~80GB for 65B model")
print("Full fine-tuning requires ~780GB for 65B model")
```

### Memory Savings Breakdown

| Model Size | Full FT | LoRA | QLoRA |
|------------|---------|------|-------|
| 7B | 28GB | 14GB | 8GB |
| 13B | 52GB | 26GB | 14GB |
| 70B | 280GB | 140GB | 48GB |

## Adapters

### Architecture

Small bottleneck modules inserted between transformer layers:

```
Input → [LayerNorm → Adapter → Residual] → Output

Adapter structure:
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
        
        # Initialize
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
        return x + residual  # Residual connection

# Adapter configuration
from peft import AdapterConfig

adapter_config = AdapterConfig(
    adapter_dim=64,          # Bottleneck dimension
    hidden_act="gelu",       # Activation function
    adapter_dropout=0.1,
    target_modules=["attention", "mlp"]
)
```

## Prompt Tuning

### Soft Prompts

Instead of discrete text prompts, learn continuous embeddings:

```python
class PromptTuning(nn.Module):
    def __init__(self, num_tokens, token_dim, initialize_from_vocab=True):
        super().__init__()
        
        # Initialize soft prompt tokens
        if initialize_from_vocab:
            # Initialize from existing vocabulary
            self.soft_prompt = nn.Parameter(
                torch.randn(num_tokens, token_dim) * 0.01
            )
        else:
            # Random initialization
            self.soft_prompt = nn.Parameter(
                torch.randn(num_tokens, token_dim)
            )
    
    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)
        
        # Expand soft prompt for batch
        soft_prompt_embeds = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Concatenate: [soft prompt] + [input]
        return torch.cat([soft_prompt_embeds, input_embeds], dim=1)

# Usage with transformer
class PromptTunedTransformer(nn.Module):
    def __init__(self, base_model, num_prompt_tokens=20):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Learnable soft prompts
        self.prompt_tuning = PromptTuning(
            num_tokens=num_prompt_tokens,
            token_dim=self.base_model.config.hidden_size
        )
    
    def forward(self, input_ids):
        # Get input embeddings
        input_embeds = self.base_model.embeddings(input_ids)
        
        # Add soft prompts
        prompted_embeds = self.prompt_tuning(input_embeds)
        
        # Pass through frozen transformer
        return self.base_model(inputs_embeds=prompted_embeds)
```

## P-tuning v2

### Deep Prompt Tuning

Add trainable prompts at every layer, not just input:

```python
class PTuningV2(nn.Module):
    def __init__(self, num_layers, num_tokens, hidden_size):
        super().__init__()
        
        # Prompt embeddings for each layer
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.01)
            for _ in range(num_layers)
        ])
        
        # MLP for prompt generation (optional)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
    
    def forward(self, hidden_states, layer_idx):
        """Add prompts to hidden states at specific layer"""
        batch_size = hidden_states.size(0)
        
        # Get prompts for this layer
        prompts = self.prompt_embeddings[layer_idx]
        prompts = self.prompt_encoder(prompts)
        
        # Expand for batch
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate
        return torch.cat([prompts, hidden_states], dim=1)
```

## Comparison

| Method | Trainable Params | Memory | Speed | Performance |
|--------|-----------------|--------|-------|-------------|
| **Full FT** | 100% | 100% | 1x | 100% |
| **LoRA** | 0.5-2% | 30-40% | 4x | 98% |
| **QLoRA** | 0.5-2% | 20-25% | 3x | 97% |
| **Adapter** | 1-3% | 35-45% | 3x | 96% |
| **Prompt Tuning** | <0.1% | 15-20% | 5x | 90% |
| **P-tuning v2** | 0.2-0.5% | 20-25% | 4x | 94% |

## Hyperparameter Tuning

### LoRA Rank Selection

| Rank | Parameters | Performance | Use Case |
|------|-----------|-------------|----------|
| 4 | 0.12% | 92% | Quick experiments |
| 8 | 0.24% | 95% | Limited resources |
| 16 | 0.47% | 98% | **Recommended** |
| 32 | 0.94% | 99% | High performance |
| 64 | 1.88% | 99.5% | Maximum quality |

### Learning Rate Guidelines

| Model Size | LoRA LR | Full FT LR |
|------------|---------|------------|
| < 1B | 1e-3 | 5e-5 |
| 7B | 1e-4 | 2e-5 |
| 13B | 1e-4 | 1e-5 |
| 70B+ | 5e-5 | 5e-6 |

## Best Practices

### 1. Module Selection

```python
# Minimal configuration (fastest)
target_modules = ["q_proj", "v_proj"]

# Recommended (best balance)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Comprehensive (highest quality)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### 2. Training Configuration

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,           # Higher than full FT
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",     # For QLoRA
    group_by_length=True,        # Efficiency
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

### 3. Inference and Deployment

```python
# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load LoRA weights
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./lora_weights")

# Option 1: Merge weights (faster inference, no PEFT dependency)
model = model.merge_and_unload()
model.save_pretrained("./merged_model")

# Option 2: Keep separate (memory efficient, easy to swap adapters)
# Just use model for inference directly
```

## Common Pitfalls

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Rank too low** | Poor performance | Increase r to 16+ |
| **Wrong modules** | No improvement | Use q_proj, v_proj at minimum |
| **Learning rate too low** | Slow convergence | Use 10-100x full FT LR |
| **Overfitting** | High train loss, low val | Add dropout, reduce epochs |
| **Instability** | Loss spikes | Gradient clipping, warmup |

---

**Previous**: [Pre-training](../pre-training/README.md) | **Next**: [Alignment](../alignment/README.md)