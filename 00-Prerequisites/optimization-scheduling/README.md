# 学习率怎么变才能又快又稳？—— 调度与梯度控制

## 这个问题从哪来

> 上一章我们学了反向传播和优化器（SGD、Adam），它们回答了"往哪走"的问题。但还有一个关键变量：**走多大的步**？
> 固定学习率有一个根本矛盾：初期需要大步快速探索，后期需要小步精细调优。如果全程用同一个学习率，要么前期太慢，要么后期震荡。
> 学习率调度就是解决这个矛盾的——让学习率在训练过程中按策略变化。

## 学习目标

完成本章后，你应能回答：

1. 为什么 Transformer 训练必须用 warmup？
2. Cosine Decay 比 Step Decay 好在哪？
3. Gradient Clipping 的按范数和按值裁剪有什么区别？

---

## 1. 直觉

学习率是"步幅"。

- **太大**：在最优解附近来回跳，永远到不了
- **太小**：收敛极慢，训练到天荒地老
- **刚好**：快速收敛到最优解附近

但"刚好"不是一个固定值——它是训练过程中不断变化的。就像登山：山脚可以大步跑（大学习率），接近山顶要小步挪（小学习率），否则会冲过头。

> 你要记住：学习率调度不是"锦上添花"，而是训练稳定性的核心组件。Transformer 的论文证明，没有 warmup 的训练直接崩溃。

---

## 2. 机制

### 2.1 为什么学习率需要变化

```
Loss
 │  ╲
 │    ╲  ← 初期：loss 很高，需要大学习率快速下降
 │      ╲
 │        ╲___
 │            ╲___ ← 后期：接近最优解，需要小学习率精细调整
 │                ───
 └──────────────────── Steps
```

固定学习率的根本问题：
- 初期梯度大、方向明确，大学习率效率高
- 后期梯度小、在最优解附近波动，大学习率导致震荡

### 2.2 学习率调度策略

**Step Decay**：每隔 N 个 epoch 乘以 γ（如 ×0.1）

```python
# 每 30 个 epoch 学习率减半
lr = base_lr * (0.5 ** (epoch // 30))
```

简单粗暴，但有"台阶"导致梯度突变——模型在台阶处突然减速，可能跳过更好的解。

**Cosine Annealing**：学习率沿余弦曲线从初始值降到最低值

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{\pi t}{T}))$$

```python
import numpy as np
import matplotlib.pyplot as plt

T = 100  # 总步数
eta_max, eta_min = 0.1, 0.001
t = np.arange(T)
lr_cosine = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))
lr_step = eta_max * (0.5 ** (t // 25))

plt.figure(figsize=(10, 4))
plt.plot(t, lr_cosine, label="Cosine Annealing")
plt.plot(t, lr_step, label="Step Decay (×0.5 per 25)")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.title("Cosine vs Step Decay")
plt.tight_layout()
plt.savefig("lr_schedules.png", dpi=150)
plt.show()
```

为什么 cosine 比 step decay 更受欢迎：
- 平滑变化，没有梯度突变
- 初期下降快，后期下降慢，符合"粗调→精调"直觉
- 配合 warmup 使用效果最佳

**Warmup**：训练初期从很小的学习率线性升到目标值。

```python
# 线性 warmup
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    lr = base_lr  # 之后由其他 scheduler 接管
```

**为什么 Transformer 需要 warmup？**

Adam 的自适应学习率统计量（一阶矩 m、二阶矩 v）在训练初期估计不准确。如果此时学习率太大，不准确的自适应缩放会导致参数更新方向偏差巨大，训练直接崩溃（loss NaN 或飞升）。

warmup 让模型在统计量稳定之前"小步走"，等估计准确了再加速。

**常见组合**：线性 warmup + cosine decay——先升后降，Transformer 训练的标准配置。

**One-Cycle Policy**：先升后降，单周期完成训练。Smith (2017) 提出，训练效率高于传统调度。

**ReduceOnPlateau**：验证 loss 停滞时自动降低学习率。适合微调场景，不需要手动设衰减策略。

### 2.3 梯度裁剪

**Gradient Clipping by Norm**：梯度向量的 L2 范数超过阈值时等比缩放。

$$\text{if } \|g\| > \text{max\_norm}: \quad g = g \times \frac{\text{max\_norm}}{\|g\|}$$

- 保持梯度方向不变，只缩放大小
- RNN/LSTM 训练的标准操作（BPTT 梯度爆炸对策）
- 常用阈值：`max_norm = 1.0`

```python
import torch.nn as nn

# PyTorch 一行搞定
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Gradient Clipping by Value**：每个梯度分量截断到 [-v, v]。

```python
nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

- 更简单但**改变了梯度方向**（不同分量被裁剪的程度不同）
- 通常不如按范数裁剪

| 方式 | 保持方向 | 适用场景 |
|------|---------|---------|
| 按范数 | 是 | NLP 任务（RNN、Transformer） |
| 按值 | 否 | 简单场景，CV 任务偶尔使用 |

### 2.4 优化器进阶

**AdamW**：权重衰减从梯度更新中解耦。

Adam + L2 正则化 ≠ 真正的权重衰减。因为 Adam 的自适应学习率会改变 L2 项的实际效果——被大幅缩放的参数，其正则化效果也被缩放了。

AdamW 把 weight decay 直接乘在权重上，不经过梯度：

```python
# Adam：L2 正则化通过梯度传递（被自适应学习率干扰）
loss = original_loss + weight_decay * (param ** 2).sum()
loss.backward()
optimizer.step()

# AdamW：权重衰减直接作用于参数（不经过梯度）
# PyTorch 的 AdamW 内部实现：
param.data = param.data * (1 - lr * weight_decay)  # 直接衰减
param.data = param.data - lr * m_hat / (sqrt(v_hat) + eps)  # 再做 Adam 更新
```

> AdamW 是 Transformer 训练的事实标准。几乎所有大语言模型预训练都用 AdamW。

**LAMB（Layer-wise Adaptive Moments）**：解决大 batch 训练的问题。

当 batch size 从 256 增加到 8192 时，Adam 会发散。LAMB 在 Adam 的基础上加了一层"层级别"的自适应缩放，让大 batch 训练也能稳定。

**Lookahead（"Lazy Adam"）**：维护两组权重——快权重（频繁更新）和慢权重（偶尔同步），减少优化器的方差。

### 2.5 训练配置速查

| 场景 | 优化器 | 调度策略 | 典型学习率 |
|------|--------|---------|-----------|
| Transformer 预训练 | AdamW | 线性 warmup (1-5% steps) + cosine decay | 1e-4 ~ 3e-4 |
| CNN 训练 | SGD + Momentum (0.9) | Step Decay | 0.1 → 0.01 → 0.001 |
| Transformer 微调 | AdamW | 短 warmup + 少量 epoch | 1e-5 ~ 5e-5 |
| LLM 微调 (LoRA) | AdamW | 无 warmup 或极短 warmup | 1e-4 ~ 3e-4 |

> 微调学习率通常是预训练的 1/10 到 1/100。太大导致灾难性遗忘。

---

## 3. 渐进式实现

**Step 1 · 手写 Cosine Decay 调度器**

```python
import numpy as np
import matplotlib.pyplot as plt

def cosine_schedule(total_steps, eta_max=0.1, eta_min=1e-4, warmup_steps=0):
    """线性 warmup + cosine decay"""
    lrs = []
    for t in range(total_steps):
        if t < warmup_steps:
            lr = eta_max * t / warmup_steps
        else:
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    return np.array(lrs)

steps = 1000
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 纯 cosine
axes[0].plot(cosine_schedule(steps))
axes[0].set_title("纯 Cosine Decay")

# warmup + cosine
axes[1].plot(cosine_schedule(steps, warmup_steps=50))
axes[1].set_title("50 步 Warmup + Cosine")

# 不同 warmup 长度
for wp in [0, 50, 100, 200]:
    axes[2].plot(cosine_schedule(steps, warmup_steps=wp), label=f"warmup={wp}")
axes[2].legend()
axes[2].set_title("不同 Warmup 长度对比")

for ax in axes:
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
plt.tight_layout()
plt.savefig("cosine_schedules.png", dpi=150)
plt.show()
```

**Step 2 · PyTorch lr_scheduler 验证**

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
optimizer = optim.AdamW(model.parameters(), lr=0.1)

# CosineAnnealingLR
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)

lrs = []
for epoch in range(100):
    lrs.append(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()

print(f"初始 LR: {lrs[0]:.4f}")
print(f"最终 LR: {lrs[-1]:.6f}")
print(f"第 50 epoch LR: {lrs[49]:.6f}")
```

**Step 3 · Gradient Clipping 实验**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 模拟一个容易梯度爆炸的 RNN
rnn = nn.RNN(10, 20, num_layers=3)
x = torch.randn(50, 1, 10)  # 长序列
h0 = torch.zeros(3, 1, 20)

# 不裁剪：手动计算梯度范数
output, hn = rnn(x, h0)
loss = output.sum()
loss.backward()
total_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in rnn.parameters() if p.grad is not None))
print(f"裁剪前梯度范数: {total_norm:.1f}")  # 可能很大（1e3~1e6）

# 裁剪后
rnn.zero_grad()
output, hn = rnn(x, h0)
loss = output.sum()
loss.backward()
grad_norm_clipped = nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
print(f"裁剪后梯度范数: {grad_norm_clipped:.1f}")  # 最大为 1.0
```

**Step 4 · AdamW vs Adam 对比**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(200, 20)
Y = torch.randn(200, 1)

def train_with_optimizer(opt_class, opt_kwargs, epochs=50):
    torch.manual_seed(42)
    m = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = opt_class(m.parameters(), **opt_kwargs)
    losses = []
    for _ in range(epochs):
        pred = m(X)
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

losses_adam = train_with_optimizer(torch.optim.Adam, {"lr": 1e-3, "weight_decay": 0.01})
losses_adamw = train_with_optimizer(torch.optim.AdamW, {"lr": 1e-3, "weight_decay": 0.01})

print(f"Adam 最终 loss:   {losses_adam[-1]:.4f}")
print(f"AdamW 最终 loss:  {losses_adamw[-1]:.4f}")
# AdamW 的权重衰减更"正确"，在正则化场景下通常效果更好
```

---

## 4. 工程陷阱（按严重度排序）

1. **warmup 步数设错导致训练崩**（最常见）
   现象：Transformer 训练前几个 step loss 直接 NaN 或飞升。
   处置：warmup steps 通常设为总步数的 1-5%，或至少 500-1000 步。

2. **scheduler.step() 调用位置错误**
   现象：在 `optimizer.step()` 之前调用 `scheduler.step()`，学习率更新时机错位。
   处置：PyTorch 标准顺序是 `loss.backward() → optimizer.step() → scheduler.step()`。

3. **gradient accumulation 与 lr schedule 的配合**
   现象：gradient accumulation 模拟大 batch，但 lr schedule 按实际 step 更新，学习率下降太快。
   处置：scheduler step 应该在 accumulation 完成后调用，或按 effective step 计算。
   → 详见 [数值精度](../numerical-precision/README.md)

4. **微调时学习率太大**
   现象：预训练模型微调几个 epoch 就灾难性遗忘。
   处置：微调学习率通常是预训练的 1/10 到 1/100（如 5e-5 vs 3e-4）。

---

## 演进笔记

> **学习率调度的遗产**：从 SGD + 手动衰减，到 Adam 的自适应学习率，再到 AdamW + warmup + cosine decay 的组合——优化策略的演进本质上是在回答"怎么用最少的计算走到最优解"。
>
> AdamW (2017) 解决了"正则化应该怎么做"的问题，成为 Transformer 训练的事实标准。Warmup 解决了"训练初期怎么不崩"的问题。Cosine decay 解决了"后期怎么精调"的问题。三者组合，几乎覆盖了所有场景。
>
> **留下的新问题**：参数更新的方向和步长都有了，但训练过程中激活值的分布会漂移——这引出了归一化机制。

→ 下一章：[归一化机制 — 为什么训练深度网络需要"校准仪"？](../normalization/README.md)

---

**上一章**：[反向传播与优化器](../backpropagation/README.md) | **下一章**：[归一化机制](../normalization/README.md)
