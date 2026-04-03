# 怎么量化"错得有多离谱"？—— 损失函数全景

## 这个问题从哪来

> 前一章我们学了 Softmax，它把模型输出变成概率。但模型怎么知道自己的预测"错得有多离谱"？这需要损失函数来量化。
> 不同任务需要不同的"评分标准"：回归任务关注"差了多少"，分类任务关注"选错了哪个类"，对比学习关注"正样本是否比负样本更近"。选错损失函数，就像用错尺子量东西——模型会学到错误的行为。

## 学习目标

完成本章后，你应能回答：

1. 回归、分类、对比学习三类损失各自的设计动机是什么？
2. Focal Loss 如何解决类别不平衡问题？
3. Label Smoothing 为什么能防止模型过度自信？

---

## 1. 直觉

损失函数是老师的"评分标准"。

- **MSE**（均方误差）像一个严厉的老师：答错越多，扣分越重（误差的平方放大了大错误）。
- **交叉熵**像一个只看"选对还是选错"的老师：它不关心你差多少分，只关心你把概率分配给了正确的类没有。
- **对比损失**像一个体育老师：它不关心绝对成绩，只关心"跑得比对手快就行"——正样本必须比负样本更相似。

> 你要记住：损失函数决定了模型优化的方向。换一个损失函数，模型可能学到完全不同的东西。

---

## 2. 机制

### 2.1 回归损失

当目标是连续值时（预测房价、温度、坐标等）。

**MSE（均方误差）**

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

特点：对大误差惩罚更重（平方放大），梯度随误差线性增长（$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$）。对异常值敏感。

**MAE（平均绝对误差）**

$$
L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

特点：对异常值更鲁棒，但梯度恒为 $\pm 1$，在零点处不可导。

**Huber Loss（两者折中）**

$$
L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}
$$

小误差时用 MSE（平滑梯度），大误差时用 MAE（限制惩罚），兼顾两者优点。

| 损失 | 对异常值 | 梯度行为 | 适用场景 |
|------|---------|---------|---------|
| MSE | 敏感 | 线性增长 | 误差分布接近高斯 |
| MAE | 鲁棒 | 恒为 ±1 | 存在异常值 |
| Huber | 折中 | 平滑过渡 | 需要两者优点 |

### 2.2 分类损失

**交叉熵（Cross-Entropy）**

多分类的标准损失。真实标签 $y$ 是 one-hot 向量，预测 $\hat{y}$ 是 softmax 输出：

$$
L_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_c
$$

其中 $c$ 是正确类别。直觉：给正确类分配的概率越高，loss 越接近 0；如果给正确类极低的概率，loss 趋向 $+\infty$。

PyTorch 中 `nn.CrossEntropyLoss()` 接收的是 logits（未经过 softmax），内部自动做 `log_softmax + NLLLoss`，比手动组合更稳定。

**二分类交叉熵（BCE）**

$$
L_{\text{BCE}} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]
$$

PyTorch 中 `nn.BCEWithLogitsLoss()` 同样接收 logits，内部用 sigmoid + log 数值稳定实现。

**Focal Loss（Lin et al., 2017）**

解决类别不平衡问题。标准交叉熵对易分样本（$\hat{y}_c$ 接近 1）的 loss 接近 0，但大量易分样本的 loss 加起来仍然不小，会淹没难分样本的信号。

$$
L_{\text{Focal}} = -(1 - \hat{y}_c)^\gamma \log \hat{y}_c
$$

$\gamma \geq 0$ 是聚焦参数：
- $\gamma = 0$：退化为标准交叉熵
- $\gamma > 0$：易分样本（$\hat{y}_c \to 1$）的权重 $(1 - \hat{y}_c)^\gamma \to 0$，被自动降低
- $\gamma = 2$ 是常用值

直觉：让模型"聚焦"在它分不好的样本上，而不是花精力在已经分对的样本上。

> 你要记住：Focal Loss 的核心是 $(1 - \hat{y}_c)^\gamma$ 这个调制因子——越容易的样本权重越低。

### 2.3 对比学习损失

对比学习不依赖标签，而是学习"相似的东西应该靠近，不相似的东西应该远离"。

**InfoNCE Loss**

$$
L = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\sum_{j=0}^{K} \exp(\text{sim}(q, k_j) / \tau)}
$$

其中 $q$ 是查询（query），$k_+$ 是正样本，$k_j$ 包含正样本和 $K-1$ 个负样本，$\tau$ 是温度参数，$\text{sim}$ 是相似度函数（通常用余弦相似度）。

本质上就是：在正样本和所有候选中，正样本的 softmax 概率要尽量高。

**对称交叉熵（CLIP 使用）**

CLIP 对图像和文本做对比学习，分别在两个方向计算 InfoNCE，然后取平均：

$$
L = \frac{1}{2}(L_{\text{image} \to \text{text}} + L_{\text{text} \to \text{image}})
$$

每个方向就是一个 softmax：给定一张图，所有文本中正确的文本概率要最高，反之亦然。

### 2.4 Label Smoothing

标准分类的目标是 one-hot：正确类为 1，其余为 0。这会鼓励模型输出极端概率（趋近 1 或 0），导致过度自信。

Label Smoothing 把硬标签变软：

$$
y_k^{\text{smooth}} = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K} & \text{if } k = c \\ \frac{\epsilon}{K} & \text{otherwise} \end{cases}
$$

$\epsilon$ 通常取 0.1。直觉：告诉模型"你对正确类只有 90% 的把握，剩下 10% 平均分给其他类"。这防止模型输出极端 logit，提高泛化能力。

> 在 Transformer 训练中几乎标配，原始 Transformer 论文就使用了 $\epsilon = 0.1$ 的 label smoothing。

---

## 3. 渐进式实现

**Step 1 · 回归损失对比**

```python
import numpy as np

y = np.array([3.0])
y_hat = np.linspace(0, 6, 100)

mse = (y - y_hat) ** 2
mae = np.abs(y - y_hat)
huber_delta = 1.0
huber = np.where(
    np.abs(y - y_hat) <= huber_delta,
    0.5 * (y - y_hat) ** 2,
    huber_delta * np.abs(y - y_hat) - 0.5 * huber_delta ** 2
)

print(f"误差=3 时: MSE={mse[-1]:.1f}, MAE={mae[-1]:.1f}, Huber={huber[-1]:.2f}")
print(f"误差=0.5 时: MSE={mse[58]:.4f}, MAE={mae[58]:.4f}")
# MSE 对大误差惩罚远大于 MAE
```

**Step 2 · 交叉熵与 Focal Loss**

```python
import numpy as np

def cross_entropy(p_correct):
    """标准交叉熵（p_correct 是给正确类的概率）"""
    return -np.log(p_correct + 1e-8)

def focal_loss(p_correct, gamma=2.0):
    """Focal Loss"""
    return -(1 - p_correct) ** gamma * np.log(p_correct + 1e-8)

probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])

print(f"{'p_correct':>10} {'CE':>8} {'Focal(g=2)':>12} {'Focal/CE':>10}")
print("-" * 44)
for p in probs:
    ce = cross_entropy(p)
    fl = focal_loss(p, gamma=2.0)
    print(f"{p:>10.2f} {ce:>8.4f} {fl:>12.4f} {fl/ce:>10.2%}")
# p_correct=0.99 时 Focal/CE 仅 0.01%，易分样本被大幅压制
# p_correct=0.1 时 Focal/CE 仍有 81%，难分样本几乎不受影响
```

**Step 3 · Label Smoothing 实现**

```python
import torch
import torch.nn as nn

def label_smoothing_manual(labels, num_classes, epsilon=0.1):
    """手动实现 label smoothing"""
    one_hot = torch.zeros(labels.size(0), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    smooth = one_hot * (1 - epsilon) + epsilon / num_classes
    return smooth

labels = torch.tensor([2])  # 正确类别是第 2 类
smooth_labels = label_smoothing_manual(labels, num_classes=5, epsilon=0.1)
print(f"原始 one-hot: [0, 0, 1, 0, 0]")
print(f"Smooth 后:    {smooth_labels[0].tolist()}")
# [0.02, 0.02, 0.92, 0.02, 0.02]

# PyTorch 内置支持
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
logits = torch.randn(1, 5)
target = torch.tensor([2])
loss = loss_fn(logits, target)
print(f"Label smoothing loss: {loss.item():.4f}")
```

**Step 4 · InfoNCE 实现与验证**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

BATCH, DIM = 4, 64
TEMPERATURE = 0.07

# 模拟查询和键
q = F.normalize(torch.randn(BATCH, DIM), dim=-1)  # queries
k = F.normalize(torch.randn(BATCH, DIM), dim=-1)  # keys

# 相似度矩阵：(batch, batch)
sim = q @ k.T / TEMPERATURE

# 对角线是正样本对（q_i 和 k_i），非对角线是负样本
labels = torch.arange(BATCH)

# 单方向 InfoNCE = CrossEntropyLoss(sim, arange(batch))
loss = nn.CrossEntropyLoss()(sim, labels)
print(f"InfoNCE loss: {loss.item():.4f}")

# 对称版本（CLIP 风格）：image→text + text→image
sim_t2i = k @ q.T / TEMPERATURE
loss_symmetric = (nn.CrossEntropyLoss()(sim, labels) + nn.CrossEntropyLoss()(sim_t2i, labels)) / 2
print(f"对称 InfoNCE loss: {loss_symmetric.item():.4f}")
```

---

## 4. 工程陷阱（按严重度排序）

1. **log(0) 导致 NaN**
   现象：交叉熵在 $\hat{y} = 0$ 时 $\log(0) = -\infty$。
   处置：永远用 `nn.CrossEntropyLoss()` 或 `nn.BCEWithLogitsLoss()`，它们内部有数值稳定实现。不要手动 `log(softmax(...))`。

2. **类别不平衡时交叉熵被多数类主导**
   现象：100 个样本中 95 个是类 A，模型只要全预测 A 就有 95% 准确率，loss 也很低。
   处置：使用 Focal Loss 降低易分样本权重，或对少数类过采样，或在 loss 中设置类别权重 `nn.CrossEntropyLoss(weight=class_weights)`。

3. **对比学习的温度参数设错**
   现象：温度 $\tau$ 太大 → softmax 趋向均匀，模型学不到区分；太小 → softmax 趋向 one-hot，梯度消失。
   处置：CLIP 用 $\tau = 0.07$，SimCLR 用 $\tau = 0.5$，可根据任务在 0.05–0.5 范围内调试。$\tau$ 也可以作为可学习参数。

4. **Label Smoothing 的 epsilon 设太大**
   现象：$\epsilon = 0.4$ 时模型学不到明确的类别信号，训练 loss 降不下去。
   处置：标准值 $\epsilon = 0.1$，一般不超过 0.2。

5. **回归任务用错了损失**
   现象：数据中有少量极端异常值，用 MSE 被异常值拉偏。
   处置：先用 Huber Loss（$\delta=1.0$），或先清洗异常值再决定用 MSE 还是 MAE。

> 你要记住：损失函数选择的关键是"你希望模型关注什么"。回归关注数值精度、分类关注类别概率、对比学习关注相对距离。

---

## 演进笔记

> **损失函数的演进脉络**：从监督学习（MSE → 交叉熵 → Focal Loss）到对比学习（InfoNCE → 对称交叉熵），再到 RLHF（Reward Model 的 ranking loss）——损失函数的设计始终在回答同一个问题："什么信号能让模型学到我们想要的行为？"
>
> Focal Loss 的"调制因子"思想后来也影响了难例挖掘（Hard Example Mining）和课程学习（Curriculum Learning）。
>
> **留下的新问题**：对比损失需要大量负样本才能学好——这催生了 MoCo 的动量编码器和 SimCLR 的大 batch size 策略。

→ 下一章：[反向传播与优化器](../backpropagation/README.md)

---

**上一章**：[Softmax 与概率分布](../softmax/README.md) | **下一章**：[反向传播与优化器](../backpropagation/README.md)
