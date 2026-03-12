[English](README_EN.md) | [中文](README.md)

# 神经网络训练

## 概述

神经网络训练是“优化问题 + 泛化问题”的组合：你不仅要让训练集 loss 降下去，还要让模型在未见数据上表现稳定。本章面向已有机器学习基础的读者，重点是建立一条可复用的训练闭环，并给出排障优先级。

## 学习目标

完成本章后，你应能回答：

1. 训练循环的关键步骤与顺序是什么？
2. 优化器、学习率调度、正则化分别影响什么？
3. 当训练异常时，先查哪三件事最有效？

## 1. 训练闭环先建立

一个完整训练周期最小包含：

1. 前向计算得到 loss
2. 反向传播得到梯度
3. 参数更新
4. 在验证集评估并记录指标
5. 保存最佳检查点

你要记住：训练工程的第一优先级不是“更复杂技巧”，而是“闭环稳定可复现”。

## 2. 优化器怎么选

### 2.1 SGD 与 Momentum

$$
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)
$$

加入动量后：

$$
v_t=\beta v_{t-1}+\nabla L(\theta_t),\quad
\theta_{t+1}=\theta_t-\eta v_t
$$

特点：泛化常较好，但对学习率更敏感。

### 2.2 Adam 与 AdamW

Adam 使用一阶/二阶矩自适应缩放梯度，收敛快、开箱即用。  
AdamW 把权重衰减与梯度更新解耦，通常是现代默认选择。

| 优化器 | 起步建议 | 适用场景 |
|--------|----------|----------|
| SGD + Momentum | `lr=0.1`（需配调度） | 大规模视觉训练、追求泛化 |
| Adam | `lr=1e-3` | 快速原型 |
| AdamW | `lr=1e-3, wd=1e-2` | 通用默认、Transformer/CNN |

你要记住：不确定时先 AdamW，目标是先跑出稳定基线。

## 3. 学习率调度决定收敛轨迹

常用策略：

- StepLR：规则简单，适合传统流程
- CosineAnnealing：平滑衰减，现代常用
- ReduceLROnPlateau：验证集停滞时自适应降 LR
- Warmup + Cosine：大模型高频配置

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
```

你要记住：学习率往往比“换模型”更影响收敛与最终精度。

## 4. 正则化与归一化

### 4.1 Dropout

- 训练时随机失活神经元，抑制共适应
- 常见范围：`0.1 ~ 0.5`
- 推理阶段自动关闭（`model.eval()`）

### 4.2 BatchNorm 与 LayerNorm

- BatchNorm：依赖 batch 统计，CNN 常用
- LayerNorm：按样本特征归一化，RNN/Transformer 常用

### 4.3 数据增强

图像任务中，增广通常是提升泛化最划算的手段之一。  
推荐最小组合：随机裁剪 + 翻转 + 归一化；再视情况加入 Mixup/CutMix。

你要记住：过拟合时先从数据增强和权重衰减入手，再考虑增大模型复杂度。

## 5. 初始化策略

| 方法 | 推荐激活函数/场景 |
|------|-------------------|
| Xavier/Glorot | Tanh/Sigmoid |
| Kaiming/He | ReLU/LeakyReLU |
| Orthogonal | RNN 循环权重 |

```python
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

你要记住：初始化出错会直接让训练“看起来像超参问题”。

## 6. 可复用训练模板（PyTorch）

```python
import torch
import torch.nn as nn
import torch.optim as optim

def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

## 7. 检查点与早停

```python
best_val = -1.0
patience, bad_epochs = 10, 0

for epoch in range(num_epochs):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
    scheduler.step()

    if val_acc > best_val:
        best_val, bad_epochs = val_acc, 0
        torch.save({"epoch": epoch, "model": model.state_dict()}, "best_model.pth")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break
```

你要记住：没有检查点的长训练，等于没有容错。

## 8. 排障优先级（先查这五项）

1. 学习率是否合理（过大震荡、过小不收敛）
2. 数据与标签是否错位（最常见隐藏错误）
3. `model.train()/eval()` 是否在正确阶段切换
4. 损失函数与输出层是否匹配（如 logits + CrossEntropy）
5. 梯度是否异常（爆炸/消失/NaN）

常见症状速查：

| 症状 | 高概率原因 | 首选动作 |
|------|------------|----------|
| loss 不降 | LR 不合适 | 做 LR 范围测试或直接降 10 倍 |
| val 远差于 train | 过拟合 | 增强 + wd + 早停 |
| loss 突升或 NaN | 数值不稳定 | 降 LR + 梯度裁剪 |
| 收敛慢 | 调度或初始化不佳 | 改 warmup/cosine + 检查初始化 |

## 9. 超参搜索建议

执行顺序建议：

1. 手工建立可靠 baseline（固定随机种子）
2. 单变量扫学习率和权重衰减
3. 再调 batch size 与模型容量
4. 最后再上随机搜索/Optuna

你要记住：超参搜索只会放大好流程，不会拯救坏流程。

## 10. 训练前检查清单

1. 训练/验证集严格隔离
2. 数据预处理在 train/val 一致且可追溯
3. 指标、日志、检查点路径已配置
4. 随机种子、环境版本、配置已记录
5. 首个 epoch 能正常跑通并保存结果

---

**上一篇**：[序列模型](../sequence-models/README.md) | **下一篇**：[注意力机制](../../03-NLP-Transformers/attention-mechanisms/README.md)
