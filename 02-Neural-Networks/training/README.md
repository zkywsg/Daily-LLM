# Neural Network Training

**[English](README.md) | [中文](README_CN.md)**

## Overview

Training deep neural networks involves optimization algorithms, regularization techniques, and careful hyperparameter tuning. This guide covers the essential components for training models effectively.

## Optimization Algorithms

### 1. Stochastic Gradient Descent (SGD)

**Basic Update Rule**:
```
θ_{t+1} = θ_t - η · ∇L(θ_t)
```

Where:
- θ: model parameters
- η: learning rate
- ∇L: gradient of loss function

**Momentum** (accelerates convergence):
```
v_t = β · v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - η · v_t
```

| Hyperparameter | Typical Range | Effect |
|---------------|---------------|---------|
| Learning rate (η) | 0.001 - 0.1 | Step size, critical for convergence |
| Momentum (β) | 0.9 - 0.99 | Smooths updates, reduces oscillation |
| Weight decay | 1e-4 - 1e-2 | L2 regularization strength |

```python
import torch
import torch.nn as nn
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

### 2. Adam (Adaptive Moment Estimation)

**Algorithm**:
```
m_t = β₁ · m_{t-1} + (1-β₁) · ∇L(θ_t)
v_t = β₂ · v_{t-1} + (1-β₂) · ∇L(θ_t)²
m̂_t = m_t / (1-β₁^t)
v̂_t = v_t / (1-β₂^t)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

| Variant | Best For | Notes |
|---------|----------|-------|
| **Adam** | General purpose | Default for many tasks |
| **AdamW** | Better regularization | Decouples weight decay |
| **Adamax** | Large embeddings | More stable for sparse gradients |

```python
# Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)

# AdamW (recommended)
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
```

### 3. Learning Rate Scheduling

| Scheduler | When to Use | Behavior |
|-----------|-------------|----------|
| **StepLR** | Standard training | Decay by factor every N epochs |
| **CosineAnnealingLR** | Modern architectures | Smooth cosine decay |
| **ReduceLROnPlateau** | Unknown optimal LR | Adaptive based on validation loss |
| **Warmup + Cosine** | Large models | Warmup then cosine decay |
| **CyclicLR** | Finding optimal LR | Cycles between bounds |

```python
# Cosine annealing with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

# Warmup scheduler
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps
)

# Main scheduler
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=1e-6
)

# Combine
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)
```

## Regularization Techniques

### 1. Dropout

**Concept**: Randomly zero out neurons during training

```
During training: Output = mask ⊙ x, where mask ~ Bernoulli(p)
During inference: Output = x · p (inverted dropout)
```

| Dropout Rate | Layer Type | Effect |
|--------------|------------|---------|
| 0.1 - 0.3 | Input layer | Slight regularization |
| 0.3 - 0.5 | Hidden layers | Standard regularization |
| 0.5 - 0.8 | Large fully connected | Heavy regularization |

```python
class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.2)   # Input dropout
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)   # Hidden dropout
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

### 2. Batch Normalization

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
- Less sensitive to initialization

**Implementation**:
```python
# 2D data (CNN)
nn.BatchNorm2d(num_features)

# 1D data (NLP, time series)
nn.BatchNorm1d(num_features)

# Usage in network
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

### 3. Layer Normalization

**When to use**: RNNs, Transformers (not batch-dependent)

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

where μ, σ² computed per sample across features
```

```python
# Transformer block with LayerNorm
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
        # Pre-norm architecture (modern)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

### 4. Data Augmentation

**Computer Vision**:
| Technique | Implementation | Effect |
|-----------|---------------|---------|
| **Random Crop** | `transforms.RandomCrop` | Translation invariance |
| **Random Flip** | `transforms.RandomHorizontalFlip` | Mirror augmentation |
| **Color Jitter** | `transforms.ColorJitter` | Color robustness |
| **AutoAugment** | `transforms.AutoAugment` | Learned augmentations |
| **Mixup/CutMix** | Custom implementation | Regularization |

```python
from torchvision import transforms

# Standard augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Mixup implementation
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
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

## Weight Initialization

### Common Strategies

| Method | Use Case | Formula |
|--------|----------|---------|
| **Xavier (Glorot)** | Tanh, Sigmoid | W ~ U[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))] |
| **Kaiming (He)** | ReLU, LeakyReLU | W ~ N(0, √(2/fan_in)) |
| **Orthogonal** | RNNs | W = QR decomposition |
| **Normal** | General | W ~ N(0, 0.02) |

```python
# Manual initialization
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

## Hyperparameter Tuning

### 1. Learning Rate Finding

**Learning Rate Range Test**:
```python
def lr_range_test(model, train_loader, optimizer, criterion, 
                  start_lr=1e-7, end_lr=10, num_iter=100):
    """Find optimal learning rate range"""
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    losses = []
    
    model.train()
    iter_loader = iter(train_loader)
    
    for lr in lrs:
        # Update learning rate
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

# Plot to find optimal range (usually steepest descent)
import matplotlib.pyplot as plt
lrs, losses = lr_range_test(model, train_loader, optimizer, criterion)
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()
```

### 2. Grid Search vs Random Search

```python
from sklearn.model_selection import ParameterGrid, ParameterSampler

# Grid Search (exhaustive but expensive)
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'dropout': [0.2, 0.5]
}

# Random Search (more efficient)
param_distributions = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128, 256],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Optuna (Bayesian optimization)
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Train model with these hyperparameters
    model = create_model(dropout=dropout)
    val_acc = train_and_validate(model, lr=lr, batch_size=batch_size)
    
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")
```

## Training Loop Best Practices

### 1. Complete Training Function

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
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

# Full training with early stopping
best_acc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    # Early stopping
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

### 2. Checkpointing

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc': best_acc,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Troubleshooting Guide

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **Loss not decreasing** | Learning rate too high/low | Use LR finder, adjust rate |
| **Validation loss >> training loss** | Overfitting | Increase dropout, add regularization, more data |
| **Training loss >> validation loss** | Underfitting | Increase model capacity, train longer |
| **Loss spikes** | Gradient explosion | Gradient clipping, reduce LR |
| **NaN loss** | Numerical instability | Check data, use gradient clipping, lower LR |
| **Very slow convergence** | Bad initialization | Use He/Xavier init, check data normalization |

## Best Practices

### 1. Optimization
- Start with Adam (lr=1e-3) or AdamW (lr=1e-3, wd=0.01)
- Use cosine LR schedule with warmup for large models
- Monitor both training and validation metrics

### 2. Regularization
- Use dropout (0.2-0.5) in fully connected layers
- Apply data augmentation appropriate for your domain
- Use early stopping to prevent overfitting

### 3. Initialization
- Use Kaiming init for ReLU networks
- Use Xavier init for tanh/sigmoid networks
- Initialize batch norm weights to 1, biases to 0

### 4. Validation
- Always use a held-out validation set
- Implement early stopping based on validation metric
- Save best model checkpoint

---

**Previous**: [CNN Architectures](../cnn-architectures/README.md) | **Next**: [Sequence Models](../sequence-models/README.md)