# LLM 预训练

[English](README_EN.md) | [中文](README.md)

## 概述

预训练是在海量文本数据上训练大型语言模型以学习通用语言表示的过程。这个基础阶段决定了模型在特定任务微调之前的能力和知识。

## 预训练目标

### 1. 因果语言建模 (Causal Language Modeling, CLM)

**GPT 风格**: 基于前文预测下一个 token

```
Context: The cat sat
Target: on
Probability: P(on | The cat sat)
```

**目标函数**:
```
L_CLM = -Σ_t log P(x_t | x_{<t}; θ)
```

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 因果 LM 预训练目标
def causal_lm_loss(model, input_ids):
    """
    标准因果语言建模损失
    """
    # 为下个 token 预测进行平移
    labels = input_ids.clone()

    # 前向传播
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    return loss

# 使用自定义损失函数实现
def compute_clm_loss(logits, targets, ignore_index=-100):
    """
    手动计算因果 LM 损失
    """
    # 平移 logits 和 targets
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()

    # 展平
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1)
    )

    return loss
```

### 2. 掩码语言建模 (Masked Language Modeling, MLM)

**BERT 风格**: 从双向上下文预测被掩码的 token

```
Input:  The [MASK] sat on the [MASK].
Target: [cat, mat]
```

**掩码策略**:
- 80%: 替换为 [MASK] token
- 10%: 替换为随机 token
- 10%: 保持原始 token

```python
def create_mlm_mask(inputs, tokenizer, mlm_prob=0.15):
    """
    创建掩码语言建模标签
    """
    labels = inputs.clone()

    # 创建概率矩阵
    prob_matrix = torch.full(labels.shape, mlm_prob)

    # 不掩码特殊 token
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    # 采样掩码索引
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # 仅在掩码 token 上计算损失

    # 80% 掩码, 10% 随机, 10% 不变
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels
```

### 3. 前缀语言建模 (Prefix Language Modeling, Prefix LM)

**T5 风格**: 在前缀上使用双向注意力,在后续上使用因果注意力

```
Input:  Translate to French: Hello world
Prefix (bidirectional): Translate to French:
Suffix (causal): Hello world
```

### 4. 混合目标

**现代方法**结合多个目标:

| Model | Primary Objective | Secondary |
|-------|------------------|-----------|
| GPT-4 | CLM | - |
| LLaMA | CLM | - |
| BERT | MLM | NSP |
| RoBERTa | MLM | - |
| T5 | Span Corruption | - |
| UL2 | Mixture of Denoisers | Multiple |

## 数据准备

### 1. 数据来源

| Source | Proportion | Examples |
|--------|-----------|----------|
| **Web Text** | 60-80% | Common Crawl, C4 |
| **Books** | 10-15% | Gutenberg, Books3 |
| **Code** | 10-20% | GitHub, StackOverflow |
| **Wikipedia** | 5-10% | Wikimedia dumps |
| **Academic** | 5% | ArXiv, PubMed |

```python
# 数据混合配置
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

### 2. 数据处理流水线

```python
import re
from typing import List, Iterator
import multiprocessing as mp

class DataProcessor:
    def __init__(self, min_length=100, max_length=100000):
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """基础文本清理"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)

        # 规范化 unicode
        text = text.strip()

        return text

    def quality_filter(self, text: str) -> bool:
        """过滤低质量文档"""
        # 长度检查
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # 字符比例检查
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return False

        # 重复检查
        lines = text.split('\n')
        if len(lines) != len(set(lines)):
            return False

        return True

    def deduplicate(self, texts: List[str]) -> List[str]:
        """移除近重复文档"""
        from datasketch import MinHashLSH, MinHash

        lsh = MinHashLSH(threshold=0.9, num_perm=128)
        unique_texts = []

        for text in texts:
            m = MinHash(num_perm=128)
            for word in text.split()[:100]:  # 采样前 100 个词
                m.update(word.encode('utf8'))

            # 检查是否已存在相似文档
            if not lsh.query(m):
                lsh.insert(text, m)
                unique_texts.append(text)

        return unique_texts

    def tokenize_batch(self, texts: List[str], tokenizer) -> Iterator[List[int]]:
        """分词并分块长文档"""
        for text in texts:
            # 分词
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # 分块为序列
            max_seq_length = 2048
            for i in range(0, len(tokens), max_seq_length):
                chunk = tokens[i:i + max_seq_length]
                if len(chunk) > 10:  # 最小长度
                    yield chunk
```

### 3. 数据格式

**常用格式**:
- **JSONL**: `{"text": "...", "metadata": {...}}`
- **Arrow**: 高效加载的列式格式
- **TFRecord/Parquet**: 大规模训练的二进制格式

```python
import pyarrow as pa
import pyarrow.parquet as pq

def save_to_parquet(examples, output_path):
    """保存分词数据到 parquet"""
    table = pa.table({
        'input_ids': pa.array(examples, type=pa.list_(pa.int64()))
    })
    pq.write_table(table, output_path)
```

## 缩放定律

### 1. Chinchilla 缩放定律

**给定计算预算下的最优模型大小**:

```
给定计算 C (以 FLOPs 为单位):
- 最优参数: N_opt ∝ C^0.50
- 最优 tokens: D_opt ∝ C^0.50

训练 FLOPs ≈ 6ND
其中:
- N: 参数数量
- D: token 数量
```

| Compute (FLOPs) | Optimal Params | Optimal Tokens |
|----------------|---------------|----------------|
| 1e18 | 400M | 8B |
| 1e19 | 1.3B | 26B |
| 1e20 | 4B | 80B |
| 1e21 | 13B | 260B |
| 1e22 | 40B | 800B |
| 1e23 | 130B | 2.6T |

### 2. 损失预测

**损失作为参数和数据的函数**:

```
L(N, D) = E + A/N^α + B/D^β

其中:
- E: 不可约熵
- A, B: 缩放系数
- α ≈ 0.34, β ≈ 0.28
```

```python
def estimate_loss(num_params, num_tokens,
                  E=1.69, A=406.4, B=410.7,
                  alpha=0.34, beta=0.28):
    """
    基于缩放定律估计预训练损失
    """
    N = num_params
    D = num_tokens

    loss = E + A / (N ** alpha) + B / (D ** beta)
    return loss

# 示例: 7B 模型使用 1T tokens
loss = estimate_loss(7e9, 1e12)
print(f"Estimated loss: {loss:.2f}")
```

### 3. 计算量估计

**训练计算公式**:

```
FLOPs ≈ 6 × N × D

其中 6N 包括:
- 2N 用于前向传播 (矩阵乘法)
- 4N 用于反向传播 (梯度)
```

```python
def estimate_training_compute(params, tokens, hardware_flops=312e12,
                               utilization=0.3, num_gpus=1024):
    """
    估计训练时间和成本
    """
    # 总 FLOPs
    total_flops = 6 * params * tokens

    # 所需 GPU 时数
    gpu_flops_per_second = hardware_flops * utilization
    total_seconds = total_flops / (gpu_flops_per_second * num_gpus)
    gpu_hours = total_seconds * num_gpus / 3600

    # 成本估计 (按 $2/GPU-hour)
    cost = gpu_hours * 2

    return {
        'total_flops': total_flops,
        'gpu_hours': gpu_hours,
        'days': total_seconds / 86400,
        'estimated_cost_usd': cost
    }

# LLaMA-2 7B 示例
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

## 预训练的分布式训练

### 1. 并行策略

| Strategy | What is Split | When to Use |
|----------|--------------|-------------|
| **Data Parallel (DP)** | 数据 batch 跨 GPU | < 1B params |
| **Tensor Parallel (TP)** | 层权重跨 GPU | 1-10B params |
| **Pipeline Parallel (PP)** | 层跨 GPU | > 10B params |
| **FSDP** | 参数、梯度、优化器状态 | 1-100B params |
| **3D Parallel** | DP + TP + PP | > 100B params |

### 2. 大模型的 PyTorch FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

def setup_fsdp_model(model, world_size):
    """
    为大模型训练设置 FSDP
    """
    # Transformer 层的自动包裹策略
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock}
    )

    # FSDP 配置
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

# 使用 FSDP 训练
for step, batch in enumerate(dataloader):
    loss = model(input_ids=batch['input_ids'])
    loss.backward()

    # 梯度裁剪
    model.clip_grad_norm_(max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    # 定期检查点
    if step % checkpoint_interval == 0:
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save(state_dict, f'checkpoint_step_{step}.pt')
```

### 3. DeepSpeed ZeRO

```python
from deepspeed import DeepSpeedEngine

# 70B 模型训练的 DeepSpeed 配置
ds_config = {
    "train_batch_size": 512,  # 全局 batch size
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
        "stage": 3,  # 最大模型使用 ZeRO-3
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

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# 训练循环
for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 4. 检查点策略

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.checkpoints = []

    def save_checkpoint(self, model, optimizer, scheduler, step, loss):
        """保存训练检查点"""
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

        # 清理旧检查点
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            old_checkpoint.unlink(missing_ok=True)

    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        """从检查点恢复"""
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

        return checkpoint['step'], checkpoint['loss']
```

## 训练稳定性

### 1. 损失尖峰

**原因和解决方案**:
| Cause | Solution |
|-------|----------|
| 坏数据点 | 梯度裁剪、数据清理 |
| 数值不稳定 | 混合精度检查、损失缩放 |
| 学习率过高 | 降低 LR、使用 warmup |
| 初始化不当 | 使用适当的权重初始化 |

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 混合精度的损失缩放
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

### 2. 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

# 余弦 warmup
num_warmup_steps = 2000
num_training_steps = 100000

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5,  # 半余弦
    min_lr_ratio=0.1  # 结束时为最大 LR 的 10%
)
```

## 最佳实践

### 1. 数据
- 积极去重
- 过滤低质量内容
- 平衡数据来源
- 监控数据分布

### 2. 训练
- 使用带 warmup 的余弦 LR 调度
- 梯度裁剪 (norm=1.0)
- 混合精度 (BF16 优于 FP16)
- 频繁检查点
- 监控损失曲线

### 3. 硬件
- < 1B 模型使用数据并行
- 1-70B 模型使用 FSDP
- > 100B 模型使用 3D 并行
- 优化通信

### 4. 评估
- 保留数据的困惑度
- 下游任务性能
- 人工评估样本

---

**上一节**: [预训练模型](../../03-NLP-Transformers/pretrained-models/README.md) | **下一节**: [PEFT](../peft/README.md)
