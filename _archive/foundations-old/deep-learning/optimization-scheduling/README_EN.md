[English](README_EN.md) | [中文](README.md)

# How Should Learning Rates Evolve for Speed and Stability? — Scheduling and Gradient Control

## Where This Problem Comes From

> In the previous chapter we learned about backpropagation and optimizers (SGD, Adam), which answer the question "which direction to go." But there is another critical variable: **how big a step to take**?
> A fixed learning rate has a fundamental contradiction: early training needs large steps for rapid exploration, while later training needs small steps for fine-tuning. Using the same learning rate throughout either makes the early stage too slow or causes late-stage oscillation.
> Learning rate scheduling resolves this contradiction by letting the learning rate change according to a strategy during training.

## Learning Objectives

After completing this chapter, you should be able to answer:

1. Why must Transformer training use warmup?
2. What makes Cosine Decay better than Step Decay?
3. What is the difference between gradient clipping by norm and gradient clipping by value?

---

## 1. Intuition

The learning rate is the "step size."

- **Too large**: bounces around the optimum and never converges
- **Too small**: converges extremely slowly, training forever
- **Just right**: quickly converges near the optimum

But "just right" is not a fixed value — it changes throughout training. Like climbing a mountain: you can take big strides at the foot (large learning rate), but need small steps near the summit (small learning rate), or you'll overshoot.

> Key takeaway: learning rate scheduling is not "icing on the cake"; it is a core component of training stability. Transformer papers prove that training without warmup collapses outright.

---

## 2. Mechanics

### 2.1 Why Learning Rates Need to Change

```
Loss
 │  ╲
 │    ╲  ← Early stage: loss is high, needs large learning rate for rapid descent
 │      ╲
 │        ╲___
 │            ╲___ ← Late stage: near optimum, needs small learning rate for fine-tuning
 │                ───
 └──────────────────── Steps
```

The fundamental problem of a fixed learning rate:
- Early stage: gradients are large and direction is clear, so a large learning rate is efficient
- Late stage: gradients are small and fluctuate near the optimum, so a large learning rate causes oscillation

### 2.2 Learning Rate Scheduling Strategies

**Step Decay**: multiply by γ (e.g., ×0.1) every N epochs

```python
# Halve the learning rate every 30 epochs
lr = base_lr * (0.5 ** (epoch // 30))
```

Simple and crude, but the "steps" cause sudden gradient changes — the model suddenly slows down at the step boundary and may skip better solutions.

**Cosine Annealing**: learning rate follows a cosine curve from the initial value down to the minimum

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{\pi t}{T}))$$

```python
import numpy as np
import matplotlib.pyplot as plt

T = 100  # Total steps
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

Why cosine is more popular than step decay:
- Smooth changes, no sudden gradient jumps
- Fast drop early, slow drop late, matching the "coarse → fine" intuition
- Best used together with warmup

**Warmup**: early in training, linearly increase the learning rate from a very small value to the target.

```python
# Linear warmup
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    lr = base_lr  # Handed over to other scheduler afterwards
```

**Why does Transformer need warmup?**

Adam's adaptive learning rate statistics (first moment m, second moment v) are inaccurate early in training. If the learning rate is too large at this time, the inaccurate adaptive scaling causes huge deviations in parameter update direction, and training collapses outright (loss becomes NaN or skyrockets).

Warmup lets the model take "small steps" until the statistics stabilize, then accelerates.

**Common combination**: linear warmup + cosine decay — first rises then falls, the standard configuration for Transformer training.

**One-Cycle Policy**: rise then fall, completing training in a single cycle. Proposed by Smith (2017), it trains more efficiently than traditional scheduling.

**ReduceOnPlateau**: automatically reduces the learning rate when validation loss plateaus. Good for fine-tuning scenarios where you don't want to manually set a decay strategy.

### 2.3 Gradient Clipping

**Gradient Clipping by Norm**: when the L2 norm of the gradient vector exceeds the threshold, scale it proportionally.

$$\text{if } \|g\| > \text{max\_norm}: \quad g = g \times \frac{\text{max\_norm}}{\|g\|}$$

- Keeps gradient direction unchanged, only scales magnitude
- Standard practice for RNN/LSTM training (countermeasure for BPTT gradient explosion)
- Common threshold: `max_norm = 1.0`

```python
import torch.nn as nn

# One-liner in PyTorch
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Gradient Clipping by Value**: clip each gradient component to [-v, v].

```python
nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

- Simpler but **changes gradient direction** (different components are clipped by different amounts)
- Usually worse than clipping by norm

| Method | Keeps direction | Applicable scenario |
|--------|----------------|---------------------|
| By norm | Yes | NLP tasks (RNN, Transformer) |
| By value | No | Simple scenarios, occasionally used in CV |

### 2.4 Advanced Optimizers

**AdamW**: decouples weight decay from the gradient update.

Adam + L2 regularization ≠ true weight decay. Because Adam's adaptive learning rate changes the actual effect of the L2 term — parameters that are heavily scaled also have their regularization effect scaled.

AdamW applies weight decay directly to the weights, bypassing the gradient:

```python
# Adam: L2 regularization passes through gradient (interfered by adaptive learning rate)
loss = original_loss + weight_decay * (param ** 2).sum()
loss.backward()
optimizer.step()

# AdamW: weight decay acts directly on parameters (bypassing gradient)
# PyTorch AdamW internal implementation:
param.data = param.data * (1 - lr * weight_decay)  # direct decay
param.data = param.data - lr * m_hat / (sqrt(v_hat) + eps)  # then Adam update
```

> AdamW is the de-facto standard for Transformer training. Almost all large language model pretraining uses AdamW.

**LAMB (Layer-wise Adaptive Moments)**: solves the problem of large-batch training.

When batch size increases from 256 to 8192, Adam diverges. LAMB adds a "layer-level" adaptive scaling on top of Adam, making large-batch training stable.

**Lookahead ("Lazy Adam")**: maintains two sets of weights — fast weights (updated frequently) and slow weights (occasionally synchronized), reducing optimizer variance.

### 2.5 Training Configuration Quick Reference

| Scenario | Optimizer | Scheduling Strategy | Typical LR |
|----------|-----------|---------------------|------------|
| Transformer pretraining | AdamW | Linear warmup (1-5% steps) + cosine decay | 1e-4 ~ 3e-4 |
| CNN training | SGD + Momentum (0.9) | Step Decay | 0.1 → 0.01 → 0.001 |
| Transformer fine-tuning | AdamW | Short warmup + few epochs | 1e-5 ~ 5e-5 |
| LLM fine-tuning (LoRA) | AdamW | No warmup or very short warmup | 1e-4 ~ 3e-4 |

> Fine-tuning learning rates are typically 1/10 to 1/100 of pretraining. Too large causes catastrophic forgetting.

---

## 3. Progressive Implementation

**Step 1 · Handwritten Cosine Decay Scheduler**

```python
import numpy as np
import matplotlib.pyplot as plt

def cosine_schedule(total_steps, eta_max=0.1, eta_min=1e-4, warmup_steps=0):
    """Linear warmup + cosine decay"""
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

# Pure cosine
axes[0].plot(cosine_schedule(steps))
axes[0].set_title("Pure Cosine Decay")

# Warmup + cosine
axes[1].plot(cosine_schedule(steps, warmup_steps=50))
axes[1].set_title("50-step Warmup + Cosine")

# Different warmup lengths
for wp in [0, 50, 100, 200]:
    axes[2].plot(cosine_schedule(steps, warmup_steps=wp), label=f"warmup={wp}")
axes[2].legend()
axes[2].set_title("Warmup Length Comparison")

for ax in axes:
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
plt.tight_layout()
plt.savefig("cosine_schedules.png", dpi=150)
plt.show()
```

**Step 2 · PyTorch lr_scheduler Verification**

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

print(f"Initial LR: {lrs[0]:.4f}")
print(f"Final LR: {lrs[-1]:.6f}")
print(f"Epoch 50 LR: {lrs[49]:.6f}")
```

**Step 3 · Gradient Clipping Experiment**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Simulate an RNN prone to gradient explosion
rnn = nn.RNN(10, 20, num_layers=3)
x = torch.randn(50, 1, 10)  # Long sequence
h0 = torch.zeros(3, 1, 20)

# Without clipping: manually compute gradient norm
output, hn = rnn(x, h0)
loss = output.sum()
loss.backward()
total_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in rnn.parameters() if p.grad is not None))
print(f"Gradient norm before clipping: {total_norm:.1f}")  # Can be very large (1e3~1e6)

# After clipping
rnn.zero_grad()
output, hn = rnn(x, h0)
loss = output.sum()
loss.backward()
grad_norm_clipped = nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
print(f"Gradient norm after clipping: {grad_norm_clipped:.1f}")  # At most 1.0
```

**Step 4 · AdamW vs Adam Comparison**

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

print(f"Adam final loss:   {losses_adam[-1]:.4f}")
print(f"AdamW final loss:  {losses_adamw[-1]:.4f}")
# AdamW's weight decay is more "correct", usually better in regularization scenarios
```

---

## 4. Engineering Pitfalls (Sorted by Severity)

1. **Wrong warmup steps causing training collapse** (most common)  
   Symptom: Transformer training loss becomes NaN or skyrockets in the first few steps.  
   Fix: warmup steps are usually set to 1-5% of total steps, or at least 500-1000 steps.

2. **`scheduler.step()` called at the wrong position**  
   Symptom: calling `scheduler.step()` before `optimizer.step()` causes misaligned learning rate updates.  
   Fix: PyTorch standard order is `loss.backward() → optimizer.step() → scheduler.step()`.

3. **Gradient accumulation combined with lr schedule**  
   Symptom: gradient accumulation simulates large batches, but the lr schedule updates by actual step, so the learning rate drops too fast.  
   Fix: scheduler step should be called after accumulation finishes, or calculated by effective step.  
   → See [Numerical Precision](../numerical-precision/README.md)

4. **Learning rate too large during fine-tuning**  
   Symptom: pretrained model forgets catastrophically after a few fine-tuning epochs.  
   Fix: fine-tuning learning rate is typically 1/10 to 1/100 of pretraining (e.g., 5e-5 vs 3e-4).

---

## Evolution Notes

> **The legacy of learning rate scheduling**: from SGD + manual decay, to Adam's adaptive learning rate, to the AdamW + warmup + cosine decay combination — the evolution of optimization strategy is essentially answering "how to reach the optimum with the least computation."
>
> AdamW (2017) solved the problem of "how should regularization be done" and became the de-facto standard for Transformer training. Warmup solved the problem of "how to not collapse early in training." Cosine decay solved the problem of "how to fine-tune later." The three combined cover almost all scenarios.
>
> **The new problem left behind**: we now have direction and step size for parameter updates, but activation distributions drift during training — this leads to normalization mechanisms.

→ Next chapter: [Normalization — Why Do Deep Networks Need a "Calibrator"?](../normalization/README.md)

---

**Previous**: [Backpropagation & Optimizers](../backpropagation/README_EN.md) | **Next**: [Activation Functions](../activation-functions/README_EN.md)
