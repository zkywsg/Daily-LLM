# LLM Pre-training

**[English](README_EN.md) | [中文](README.md)**

## Overview

Pre-training is the process of training large language models on vast amounts of text data to learn general language representations. This foundational phase determines the model's capabilities and knowledge before task-specific fine-tuning.

## Pre-training Objectives

### 1. Causal Language Modeling (CLM)

**GPT-style**: Predict next token given previous tokens

```
Context: The cat sat
Target: on
Probability: P(on | The cat sat)
```

**Objective Function**:
```
L_CLM = -Σ_t log P(x_t | x_{<t}; θ)
```

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Causal LM pre-training objective
def causal_lm_loss(model, input_ids):
    """
    Standard causal language modeling loss
    """
    # Shift for next-token prediction
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    
    return loss

# Implementation with custom loss
def compute_clm_loss(logits, targets, ignore_index=-100):
    """
    Compute causal LM loss manually
    """
    # Shift logits and targets
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    
    # Flatten
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1)
    )
    
    return loss
```

### 2. Masked Language Modeling (MLM)

**BERT-style**: Predict masked tokens from bidirectional context

```
Input:  The [MASK] sat on the [MASK].
Target: [cat, mat]
```

**Masking Strategy**:
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep original token

```python
def create_mlm_mask(inputs, tokenizer, mlm_prob=0.15):
    """
    Create masked language modeling labels
    """
    labels = inputs.clone()
    
    # Create probability matrix
    prob_matrix = torch.full(labels.shape, mlm_prob)
    
    # Don't mask special tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
        for val in labels.tolist()
    ]
    prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # Sample masked indices
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # 80% mask, 10% random, 10% unchanged
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    return inputs, labels
```

### 3. Prefix Language Modeling (Prefix LM)

**T5-style**: Use bidirectional attention on prefix, causal on suffix

```
Input:  Translate to French: Hello world
Prefix (bidirectional): Translate to French: 
Suffix (causal): Hello world
```

### 4. Mixture of Objectives

**Modern approaches** combine multiple objectives:

| Model | Primary Objective | Secondary |
|-------|------------------|-----------|
| GPT-4 | CLM | - |
| LLaMA | CLM | - |
| BERT | MLM | NSP |
| RoBERTa | MLM | - |
| T5 | Span Corruption | - |
| UL2 | Mixture of Denoisers | Multiple |

## Data Preparation

### 1. Data Sources

| Source | Proportion | Examples |
|--------|-----------|----------|
| **Web Text** | 60-80% | Common Crawl, C4 |
| **Books** | 10-15% | Gutenberg, Books3 |
| **Code** | 10-20% | GitHub, StackOverflow |
| **Wikipedia** | 5-10% | Wikimedia dumps |
| **Academic** | 5% | ArXiv, PubMed |

```python
# Data mixing configuration
data_weights = {
    'common_crawl': 0.67,
    'c4': 0.15,
    'github': 0.045,
    'wikipedia': 0.045,
    'books': 0.045,
    'arxiv': 0.025,
    'stackexchange': 0.02
}
```

### 2. Data Processing Pipeline

```python
import re
from typing import List, Iterator
import multiprocessing as mp

class DataProcessor:
    def __init__(self, min_length=100, max_length=100000):
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # Normalize unicode
        text = text.strip()
        
        return text
    
    def quality_filter(self, text: str) -> bool:
        """Filter low-quality documents"""
        # Length check
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Character ratio checks
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return False
        
        # Repetition check
        lines = text.split('\n')
        if len(lines) != len(set(lines)):
            return False
        
        return True
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate documents"""
        from datasketch import MinHashLSH, MinHash
        
        lsh = MinHashLSH(threshold=0.9, num_perm=128)
        unique_texts = []
        
        for text in texts:
            m = MinHash(num_perm=128)
            for word in text.split()[:100]:  # Sample first 100 words
                m.update(word.encode('utf8'))
            
            # Check if similar document exists
            if not lsh.query(m):
                lsh.insert(text, m)
                unique_texts.append(text)
        
        return unique_texts
    
    def tokenize_batch(self, texts: List[str], tokenizer) -> Iterator[List[int]]:
        """Tokenize with chunking for long documents"""
        for text in texts:
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Chunk into sequences
            max_seq_length = 2048
            for i in range(0, len(tokens), max_seq_length):
                chunk = tokens[i:i + max_seq_length]
                if len(chunk) > 10:  # Minimum length
                    yield chunk
```

### 3. Data Format

**Common formats**:
- **JSONL**: `{"text": "...", "metadata": {...}}`
- **Arrow**: Columnar format for efficient loading
- **TFRecord/Parquet**: Binary formats for large-scale training

```python
import pyarrow as pa
import pyarrow.parquet as pq

def save_to_parquet(examples, output_path):
    """Save tokenized data to parquet"""
    table = pa.table({
        'input_ids': pa.array(examples, type=pa.list_(pa.int64()))
    })
    pq.write_table(table, output_path)
```

## Scaling Laws

### 1. Chinchilla Scaling Laws

**Optimal model size given compute budget**:

```
Given compute C (in FLOPs):
- Optimal parameters: N_opt ∝ C^0.50
- Optimal tokens: D_opt ∝ C^0.50

Training FLOPs ≈ 6ND
Where:
- N: number of parameters
- D: number of tokens
```

| Compute (FLOPs) | Optimal Params | Optimal Tokens |
|----------------|---------------|----------------|
| 1e18 | 400M | 8B |
| 1e19 | 1.3B | 26B |
| 1e20 | 4B | 80B |
| 1e21 | 13B | 260B |
| 1e22 | 40B | 800B |
| 1e23 | 130B | 2.6T |

### 2. Loss Prediction

**Loss as function of parameters and data**:

```
L(N, D) = E + A/N^α + B/D^β

Where:
- E: irreducible entropy
- A, B: scaling coefficients
- α ≈ 0.34, β ≈ 0.28
```

```python
def estimate_loss(num_params, num_tokens, 
                  E=1.69, A=406.4, B=410.7, 
                  alpha=0.34, beta=0.28):
    """
    Estimate pre-training loss based on scaling laws
    """
    N = num_params
    D = num_tokens
    
    loss = E + A / (N ** alpha) + B / (D ** beta)
    return loss

# Example: 7B model with 1T tokens
loss = estimate_loss(7e9, 1e12)
print(f"Estimated loss: {loss:.2f}")
```

### 3. Compute Estimation

**Training compute formula**:

```
FLOPs ≈ 6 × N × D

Where 6N accounts for:
- 2N for forward pass (matrix multiplies)
- 4N for backward pass (gradients)
```

```python
def estimate_training_compute(params, tokens, hardware_flops=312e12, 
                               utilization=0.3, num_gpus=1024):
    """
    Estimate training time and cost
    """
    # Total FLOPs
    total_flops = 6 * params * tokens
    
    # GPU-hours needed
    gpu_flops_per_second = hardware_flops * utilization
    total_seconds = total_flops / (gpu_flops_per_second * num_gpus)
    gpu_hours = total_seconds * num_gpus / 3600
    
    # Cost estimation (at $2/GPU-hour)
    cost = gpu_hours * 2
    
    return {
        'total_flops': total_flops,
        'gpu_hours': gpu_hours,
        'days': total_seconds / 86400,
        'estimated_cost_usd': cost
    }

# LLaMA-2 7B example
result = estimate_training_compute(
    params=7e9,
    tokens=2e12,
    hardware_flops=312e12,  # A100
    utilization=0.3,
    num_gpus=1024
)
print(f"Training time: {result['days']:.1f} days")
print(f"GPU-hours: {result['gpu_hours']:,.0f}")
print(f"Estimated cost: ${result['estimated_cost_usd']:,.0f}")
```

## Distributed Training for Pre-training

### 1. Parallelism Strategies

| Strategy | What is Split | When to Use |
|----------|--------------|-------------|
| **Data Parallel (DP)** | Data batch across GPUs | < 1B params |
| **Tensor Parallel (TP)** | Layer weights across GPUs | 1-10B params |
| **Pipeline Parallel (PP)** | Layers across GPUs | > 10B params |
| **FSDP** | Parameters, gradients, optimizer states | 1-100B params |
| **3D Parallel** | DP + TP + PP | > 100B params |

### 2. PyTorch FSDP for Large Models

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

def setup_fsdp_model(model, world_size):
    """
    Setup FSDP for large model training
    """
    # Auto-wrap policy for transformer layers
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock}
    )
    
    # FSDP configuration
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.bfloat16,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        forward_prefetch=True,
        backward_prefetch=True,
    )
    
    return model

# Training with FSDP
for step, batch in enumerate(dataloader):
    loss = model(input_ids=batch['input_ids'])
    loss.backward()
    
    # Gradient clipping
    model.clip_grad_norm_(max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Checkpoint periodically
    if step % checkpoint_interval == 0:
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save(state_dict, f'checkpoint_step_{step}.pt')
```

### 3. DeepSpeed ZeRO

```python
from deepspeed import DeepSpeedEngine

# DeepSpeed configuration for 70B model training
ds_config = {
    "train_batch_size": 512,  # Global batch size
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 100000
        }
    },
    
    "zero_optimization": {
        "stage": 3,  # ZeRO-3 for largest models
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    
    "gradient_clipping": 1.0,
    
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16
    },
    
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop
for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 4. Checkpointing Strategy

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.checkpoints = []
    
    def save_checkpoint(self, model, optimizer, scheduler, step, loss):
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'rng_state': torch.get_rng_state(),
        }
        
        if torch.distributed.is_initialized():
            checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}_rank_{dist.get_rank()}.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # Clean old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            old_checkpoint.unlink(missing_ok=True)
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        """Resume from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        
        return checkpoint['step'], checkpoint['loss']
```

## Training Stability

### 1. Loss Spikes

**Causes and Solutions**:
| Cause | Solution |
|-------|----------|
| Bad data points | Gradient clipping, data cleaning |
| Numerical instability | Mixed precision check, loss scaling |
| Learning rate too high | Reduce LR, use warmup |
| Bad initialization | Use proper weight init |

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Loss scaling for mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss = model(batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

### 2. Learning Rate Schedule

```python
from transformers import get_cosine_schedule_with_warmup

# Cosine with linear warmup
num_warmup_steps = 2000
num_training_steps = 100000

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5,  # Half cosine
    min_lr_ratio=0.1  # End at 10% of max LR
)
```

## Best Practices

### 1. Data
- Deduplicate aggressively
- Filter low-quality content
- Balance data sources
- Monitor data distribution

### 2. Training
- Use cosine LR schedule with warmup
- Gradient clipping (norm=1.0)
- Mixed precision (BF16 preferred over FP16)
- Checkpoint frequently
- Monitor loss curves

### 3. Hardware
- Start with data parallel for < 1B
- Use FSDP for 1-70B
- Use 3D parallel for > 100B
- Optimize communication

### 4. Evaluation
- Perplexity on held-out data
- Downstream task performance
- Human evaluation samples

---

**Previous**: [Pre-trained Models](../../03-NLP-Transformers/pretrained-models/README.md) | **Next**: [PEFT](../peft/README.md)