# 分布式训练

[English](README_EN.md) | [中文](README.md)

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

### 1.1 为什么需要分布式训练？

- **模型规模**: GPT-3有175B参数，单卡无法容纳
- **数据规模**: 训练数据达到TB级别
- **时间成本**: 单机训练需要数月

### 1.2 分布式训练优势

- **加速**: 多卡并行减少训练时间
- **内存**: 分散存储大模型
- **扩展**: 支持更大模型和数据

---

## 2. 核心概念

### 2.1 并行策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **DP** | 数据并行，每卡完整模型 | 中小模型 |
| **DDP** | 分布式数据并行，高效通信 | 通用 |
| **TP** | 张量并行，层内分割 | 大模型 |
| **PP** | 流水线并行，层间分割 | 超大模型 |
| **FSDP** | 完全分片数据并行 | 大模型 |

### 2.2 通信模式

- **All-Reduce**: 梯度聚合
- **All-Gather**: 参数收集
- **Broadcast**: 广播参数

---

## 3. 数学原理

### 3.1 加速比

**Amdahl定律**:
$$
S = \frac{1}{(1-P) + \frac{P}{N}}
$$

其中:
- $S$: 加速比
- $P$: 可并行比例
- $N$: 并行单元数

### 3.2 通信开销

$$
T_{\text{total}} = T_{\text{compute}} + T_{\text{comm}} = \frac{T_{\text{single}}}{N} + \alpha + \beta \cdot M
$$

其中:
- $\alpha$: 延迟
- $\beta$: 带宽倒数
- $M$: 消息大小

---

## 4. 代码实现

### 4.1 PyTorch DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_ddp():
    """DDP训练示例"""
    local_rank = setup_distributed()

    # 创建模型
    model = MyModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=DistributedSampler(dataset)
    )

    # 训练循环
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.cuda(local_rank)
            labels = labels.cuda(local_rank)

            # 前向
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 只在主进程打印
            if local_rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 启动命令: torchrun --nproc_per_node=4 train.py
```

### 4.2 DeepSpeed ZeRO

```python
from deepspeed import DeepSpeedEngine

# DeepSpeed配置
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

# 初始化
model_engine, optimizer, _, _ = DeepSpeedEngine(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# 训练
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

---

## 5. 实验对比

### 5.1 不同并行策略对比

| GPU数 | DP加速 | DDP加速 | FSDP加速 |
|-------|--------|---------|----------|
| 2 | 1.8x | 1.9x | 1.9x |
| 4 | 3.2x | 3.7x | 3.8x |
| 8 | 5.5x | 7.2x | 7.5x |
| 16 | 8x | 13x | 14x |

### 5.2 显存节省

| 模型大小 | 单卡显存 | ZeRO-2 | ZeRO-3 |
|---------|---------|--------|--------|
| 7B | 28GB | 14GB | 7GB |
| 13B | 52GB | 26GB | 13GB |
| 70B | 280GB | 140GB | 70GB |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **选择合适的并行策略**: 小模型DDP，大模型FSDP
2. **梯度累积**: 模拟大batch size
3. **混合精度**: FP16/BF16加速训练
4. **检查点**: 定期保存防止丢失
5. **监控**: 跟踪GPU利用率和通信开销

### 6.2 常见陷阱

1. **通信瓶颈**: 忽视通信开销导致加速比低
2. **负载不均**: 数据分配不均
3. **死锁**: 不当的同步操作
4. **显存碎片**: 不管理显存导致OOM

---

## 7. 总结

分布式训练是训练大模型的必备技术：

1. **并行策略**: DP/DDP/TP/PP/FSDP选择
2. **通信优化**: 减少通信开销
3. **显存优化**: ZeRO分片节省显存
4. **扩展性**: 线性加速到数十上百卡

**选择指南**:
- < 1B参数: DDP
- 1-10B参数: FSDP
- > 10B参数: 3D并行 (TP+PP+DP)
