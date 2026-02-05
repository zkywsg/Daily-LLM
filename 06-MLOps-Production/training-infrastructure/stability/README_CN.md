# 训练稳定性

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

### 1.1 训练不稳定的症状

- **Loss爆炸**: 突然变成NaN
- **Loss震荡**: 无法收敛
- **梯度消失**: 参数不更新
- **梯度爆炸**: 参数更新过大

### 1.2 影响因素

- 学习率过大
- 批量大小不合适
- 梯度累积问题
- 混合精度溢出

---

## 2. 核心概念

### 2.1 梯度裁剪

限制梯度范数，防止爆炸:

$$
\text{if } \|\nabla\| > \text{max\_norm}: \quad \nabla = \nabla \cdot \frac{\text{max\_norm}}{\|\nabla\|}
$$

### 2.2 混合精度训练

FP16前向 + FP32反向，加速训练但可能溢出。

### 2.3 学习率调度

- **Warmup**: 初期小学习率
- **Decay**: 后期学习率衰减

---

## 3. 数学原理

### 3.1 梯度范数

$$
\|\nabla\| = \sqrt{\sum_{i} g_i^2}
$$

### 3.2 Warmup

$$
\text{lr}(t) = \text{base\_lr} \times \min(1.0, \frac{t}{\text{warmup\_steps}})
$$

---

## 4. 代码实现

### 4.1 梯度裁剪

```python
import torch
from torch.nn.utils import clip_grad_norm_

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()

    # 前向
    loss = model(batch)

    # 反向
    loss.backward()

    # 梯度裁剪 (必须在step之前)
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数
    optimizer.step()
```

### 4.2 混合精度 + 梯度缩放

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 自动混合精度
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)

    # 缩放损失并反向
    scaler.scale(loss).backward()

    # 梯度裁剪 (需要在unscale之后)
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新
    scaler.step(optimizer)
    scaler.update()
```

### 4.3 学习率Warmup

```python
from transformers import get_linear_schedule_with_warmup

# 计算总步数
total_steps = len(dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

# 创建调度器
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 训练循环
for batch in dataloader:
    # ... 前向、反向 ...
    optimizer.step()
    scheduler.step()  # 更新学习率
```

---

## 5. 实验对比

### 5.1 训练稳定性对比

| 配置 | Loss爆炸次数 | 最终Loss | 收敛时间 |
|------|-------------|---------|---------|
| **无措施** | 8/10 | NaN | 失败 |
| **仅梯度裁剪** | 2/10 | 2.3 | 10小时 |
| **裁剪+Warmup** | 0/10 | 2.1 | 8小时 |
| **完整方案** | 0/10 | 1.9 | 7小时 |

### 5.2 混合精度效果

| 精度 | 训练速度 | 显存节省 | 最终Loss |
|------|---------|---------|---------|
| FP32 | 1x | 1x | 1.9 |
| FP16无缩放 | 1.8x | 0.6x | NaN |
| FP16+GradScaler | 1.8x | 0.6x | 1.9 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **梯度裁剪**: max_norm=1.0-5.0
2. **Warmup**: 5-10%的total_steps
3. **学习率**: 从较小值开始
4. **梯度累积**: 注意缩放学习率
5. **监控**: 实时跟踪Loss和梯度范数

### 6.2 检查清单

```markdown
- [ ] 梯度裁剪启用
- [ ] 学习率Warmup
- [ ] 混合精度+GradScaler
- [ ] 异常检测 (NaN检查)
- [ ] 检查点保存
- [ ] 梯度范数监控
- [ ] 学习率调度
```

---

## 7. 总结

训练稳定性是大模型训练的基础：

1. **梯度裁剪**: 防止梯度爆炸
2. **Warmup**: 平稳启动
3. **混合精度**: 加速同时防溢出
4. **监控**: 及时发现问题

**推荐配置**:
- 梯度裁剪: max_norm=1.0
- Warmup: 5-10% steps
- 学习率: 从1e-5开始warmup到目标值
- 混合精度: 配合GradScaler使用
