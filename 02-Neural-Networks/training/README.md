[English](README_EN.md) | [中文](README.md)

# 神经网络训练

## 概述

训练深度神经网络涉及优化算法、正则化技术和精细的超参数调优。本指南涵盖了有效训练模型的关键组件。

## 优化算法

### 1. 随机梯度下降（SGD）

**基本更新规则**：
```
θ_{t+1} = θ_t - η · ∇L(θ_t)
```

其中：
- θ：模型参数
- η：学习率
- ∇L：损失函数的梯度

**动量（Momentum）**（加速收敛）：
```
v_t = β · v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - η · v_t
```

| 超参数 | 典型范围 | 效果 |
|---------------|---------------|---------|
| 学习率（η） | 0.001 - 0.1 | 步长，对收敛至关重要 |
| 动量（β） | 0.9 - 0.99 | 平滑更新，减少震荡 |
| 权重衰减 | 1e-4 - 1e-2 | L2 正则化强度 |

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 带动量的 SGD
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

### 2. Adam（自适应矩估计）

**算法**：
```
m_t = β₁ · m_{t-1} + (1-β₁) · ∇L(θ_t)
v_t = β₂ · v_{t-1} + (1-β₂) · ∇L(θ_t)²
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

| 变体 | 最适合 | 说明 |
|---------|----------|-------|
| **Adam** | 通用场景 | 多数任务的默认选择 |
| **AdamW** | 更好的正则化 | 解耦权重衰减 |
| **Adamax** | 大规模嵌入 | 对稀疏梯度更稳定 |

```python
# Adam 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)

# AdamW（推荐）
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
```

### 3. 学习率调度

| 调度器 | 使用场景 | 行为 |
|-----------|-------------|----------|
| **StepLR** | 标准训练 | 每 N 个 epoch 按因子衰减 |
| **CosineAnnealingLR** | 现代架构 | 平滑余弦衰减 |
| **ReduceLROnPlateau** | 不确定最优学习率 | 基于验证损失自适应调整 |
| **Warmup + Cosine** | 大模型 | 先预热再余弦衰减 |
| **CyclicLR** | 寻找最优学习率 | 在边界之间循环 |

```python
# 带预热的余弦退火
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

# 预热调度器
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps
)

# 主调度器
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=1e-6
)

# 组合
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)
```

## 正则化技术

### 1. Dropout

**概念**：训练期间随机将神经元置零

```
训练时：Output = mask ⊙ x，其中 mask ~ Bernoulli(p)
推理时：Output = x · p（反向 Dropout）
```

| Dropout 率 | 层类型 | 效果 |
|--------------|------------|---------|
| 0.1 - 0.3 | 输入层 | 轻微正则化 |
| 0.3 - 0.5 | 隐藏层 | 标准正则化 |
| 0.5 - 0.8 | 大型全连接层 | 强正则化 |

```python
class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.2)   # 输入层 Dropout
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)   # 隐藏层 Dropout
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

### 2. 批归一化（Batch Normalization）

**优势**：
- 减少内部协变量偏移
- 允许更高的学习率
- 充当正则化
- 对初始化不敏感

**实现**：
```python
# 二维数据（CNN）
nn.BatchNorm2d(num_features)

# 一维数据（NLP、时间序列）
nn.BatchNorm1d(num_features)

# 网络中的使用
class NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x
```

### 3. 层归一化（Layer Normalization）

**使用场景**：RNN、Transformer（不依赖批次）

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

其中 μ, σ² 在特征维度上按样本计算
```

```python
# 带有 LayerNorm 的 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm 架构（现代）
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

### 4. 数据增强

**计算机视觉**：
| 技术 | 实现 | 效果 |
|-----------|---------------|---------|
| **随机裁剪** | `transforms.RandomCrop` | 平移不变性 |
| **随机翻转** | `transforms.RandomHorizontalFlip` | 镜像增强 |
| **颜色抖动** | `transforms.ColorJitter` | 颜色鲁棒性 |
| **自动增强** | `transforms.AutoAugment` | 学习到的增强 |
| **Mixup/CutMix** | 自定义实现 | 正则化 |

```python
from torchvision import transforms

# 标准增强流水线
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Mixup 实现
def mixup_data(x, y, alpha=1.0):
    """返回混合输入、目标对和 lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## 权重初始化

### 常用策略

| 方法 | 使用场景 | 公式 |
|--------|----------|---------|
| **Xavier (Glorot)** | Tanh, Sigmoid | W ~ U[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))] |
| **Kaiming (He)** | ReLU, LeakyReLU | W ~ N(0, √(2/fan_in)) |
| **正交** | RNN | W = QR 分解 |
| **正态** | 通用 | W ~ N(0, 0.02) |

```python
# 手动初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

## 超参数调优

### 1. 学习率查找

**学习率范围测试**：
```python
def lr_range_test(model, train_loader, optimizer, criterion,
                  start_lr=1e-7, end_lr=10, num_iter=100):
    """查找最优学习率范围"""
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    losses = []

    model.train()
    iter_loader = iter(train_loader)

    for lr in lrs:
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        try:
            inputs, targets = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            inputs, targets = next(iter_loader)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return lrs, losses

# 绘图以找到最优范围（通常是最陡下降处）
import matplotlib.pyplot as plt
lrs, losses = lr_range_test(model, train_loader, optimizer, criterion)
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()
```

### 2. 网格搜索 vs 随机搜索

```python
from sklearn.model_selection import ParameterGrid, ParameterSampler

# 网格搜索（穷举但昂贵）
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'dropout': [0.2, 0.5]
}

# 随机搜索（更高效）
param_distributions = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128, 256],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Optuna（贝叶斯优化）
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # 使用这些超参数训练模型
    model = create_model(dropout=dropout)
    val_acc = train_and_validate(model, lr=lr, batch_size=batch_size)

    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")
```

## 训练循环最佳实践

### 1. 完整训练函数

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(val_loader), 100. * correct / total

# 带早停的完整训练
best_acc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    # 早停
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

### 2. 检查点保存

```python
# 保存检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc': best_acc,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 故障排查指南

| 症状 | 可能原因 | 解决方案 |
|---------|--------------|----------|
| **损失不下降** | 学习率过高/过低 | 使用 LR 查找器，调整学习率 |
| **验证损失 >> 训练损失** | 过拟合 | 增加 Dropout，添加正则化，更多数据 |
| **训练损失 >> 验证损失** | 欠拟合 | 增加模型容量，训练更长时间 |
| **损失突然上升** | 梯度爆炸 | 梯度裁剪，降低学习率 |
| **损失为 NaN** | 数值不稳定 | 检查数据，使用梯度裁剪，降低学习率 |
| **收敛很慢** | 初始化不当 | 使用 He/Xavier 初始化，检查数据归一化 |

## 最佳实践

### 1. 优化
- 从 Adam (lr=1e-3) 或 AdamW (lr=1e-3, wd=0.01) 开始
- 大模型使用带预热的余弦学习率调度
- 同时监控训练和验证指标

### 2. 正则化
- 在全连接层使用 Dropout (0.2-0.5)
- 应用适合您领域的数据增强
- 使用早停防止过拟合

### 3. 初始化
- ReLU 网络使用 Kaiming 初始化
- tanh/sigmoid 网络使用 Xavier 初始化
- 批归一化权重初始化为 1，偏置为 0

### 4. 验证
- 始终使用预留的验证集
- 基于验证指标实现早停
- 保存最佳模型检查点

---

**上一篇**：[CNN 架构](../cnn-architectures/README.md) | **下一篇**：[序列模型](../sequence-models/README.md)
