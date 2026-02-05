# Distributed Training

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

### 1.1 Why Do We Need Distributed Training?

- **Model Scale**: GPT-3 has 175B parameters, which cannot fit on a single GPU
- **Data Scale**: Training data reaches the TB level
- **Time Cost**: Single-machine training takes months

### 1.2 Advantages of Distributed Training

- **Acceleration**: Multi-GPU parallelism reduces training time
- **Memory**: Distributed storage of large models
- **Scalability**: Support for larger models and data

---

## 2. Core Concepts

### 2.1 Parallelism Strategies

| Strategy | Description | Use Case |
|----------|-------------|-----------|
| **DP** | Data Parallelism, complete model on each GPU | Small to medium models |
| **DDP** | Distributed Data Parallelism, efficient communication | General purpose |
| **TP** | Tensor Parallelism, intra-layer partitioning | Large models |
| **PP** | Pipeline Parallelism, inter-layer partitioning | Very large models |
| **FSDP** | Fully Sharded Data Parallelism | Large models |

### 2.2 Communication Patterns

- **All-Reduce**: Gradient aggregation
- **All-Gather**: Parameter collection
- **Broadcast**: Parameter broadcasting

---

## 3. Mathematical Principles

### 3.1 Speedup

**Amdahl's Law**:
$$
S = \frac{1}{(1-P) + \frac{P}{N}}
$$

Where:
- $S$: Speedup
- $P$: Parallelizable proportion
- $N$: Number of parallel units

### 3.2 Communication Overhead

$$
T_{\text{total}} = T_{\text{compute}} + T_{\text{comm}} = \frac{T_{\text{single}}}{N} + \alpha + \beta \cdot M
$$

Where:
- $\alpha$: Latency
- $\beta$: Bandwidth inverse
- $M$: Message size

---

## 4. Code Implementation

### 4.1 PyTorch DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initialize distributed environment"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_ddp():
    """DDP training example"""
    local_rank = setup_distributed()

    # Create model
    model = MyModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Data loader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=DistributedSampler(dataset)
    )

    # Training loop
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.cuda(local_rank)
            labels = labels.cuda(local_rank)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print only on main process
            if local_rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Launch command: torchrun --nproc_per_node=4 train.py
```

### 4.2 DeepSpeed ZeRO

```python
from deepspeed import DeepSpeedEngine

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4}
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO stage 2
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "fp16": {"enabled": True}
}

# Initialize
model_engine, optimizer, _, _ = DeepSpeedEngine(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

---

## 5. Experimental Comparison

### 5.1 Comparison of Different Parallelism Strategies

| GPU Count | DP Speedup | DDP Speedup | FSDP Speedup |
|------------|-------------|--------------|----------------|
| 2 | 1.8x | 1.9x | 1.9x |
| 4 | 3.2x | 3.7x | 3.8x |
| 8 | 5.5x | 7.2x | 7.5x |
| 16 | 8x | 13x | 14x |

### 5.2 Memory Savings

| Model Size | Single GPU Memory | ZeRO-2 | ZeRO-3 |
|------------|-------------------|----------|----------|
| 7B | 28GB | 14GB | 7GB |
| 13B | 52GB | 26GB | 13GB |
| 70B | 280GB | 140GB | 70GB |

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

1. **Choose appropriate parallelism strategy**: DDP for small models, FSDP for large models
2. **Gradient accumulation**: Simulate large batch size
3. **Mixed precision**: FP16/BF16 to accelerate training
4. **Checkpoints**: Save regularly to prevent loss
5. **Monitoring**: Track GPU utilization and communication overhead

### 6.2 Common Pitfalls

1. **Communication bottleneck**: Ignoring communication overhead leads to low speedup
2. **Load imbalance**: Uneven data distribution
3. **Deadlock**: Improper synchronization operations
4. **Memory fragmentation**: Not managing memory leads to OOM

---

## 7. Summary

Distributed training is essential for training large models:

1. **Parallelism strategies**: Choose between DP/DDP/TP/PP/FSDP
2. **Communication optimization**: Reduce communication overhead
3. **Memory optimization**: ZeRO sharding saves memory
4. **Scalability**: Linear speedup to dozens or hundreds of GPUs

**Selection Guide**:
- < 1B parameters: DDP
- 1-10B parameters: FSDP
- > 10B parameters: 3D parallelism (TP+PP+DP)
