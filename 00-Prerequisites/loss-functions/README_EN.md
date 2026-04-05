[English](README_EN.md) | [中文](README.md)

# How Do We Quantify "How Wrong"? — A Panorama of Loss Functions

## Where This Problem Comes From

> In the previous chapter we learned Softmax, which turns model outputs into probabilities. But how does the model know "how wrong" its predictions are? This requires a loss function to quantify the error.
> Different tasks need different "scoring criteria": regression tasks care about "how far off," classification tasks care about "which class was wrong," and contrastive learning cares about "whether positive samples are closer than negative ones." Choosing the wrong loss function is like measuring with the wrong ruler — the model will learn the wrong behavior.

## Learning Objectives

After completing this chapter, you should be able to answer:

1. What are the design motivations behind regression, classification, and contrastive learning losses?
2. How does Focal Loss solve class imbalance?
3. Why does Label Smoothing prevent the model from becoming overconfident?

---

## 1. Intuition

A loss function is the teacher's "grading criterion."

- **MSE** (mean squared error) is like a strict teacher: the more wrong you are, the heavier the penalty (squaring the error magnifies large mistakes).
- **Cross-entropy** is like a teacher who only cares "did you pick right or wrong": it doesn't care about the margin, only whether probability was assigned to the correct class.
- **Contrastive loss** is like a gym teacher: it doesn't care about absolute scores, only "did you run faster than your opponent" — positive samples must be more similar than negative ones.

> Key takeaway: the loss function determines the direction of model optimization. Change the loss function, and the model may learn something completely different.

---

## 2. Mechanics

### 2.1 Regression Losses

When the target is a continuous value (predicting house prices, temperature, coordinates, etc.).

**MSE (Mean Squared Error)**

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Characteristics: heavier penalty for large errors (square magnifies them), gradient grows linearly with error ($\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$). Sensitive to outliers.

**MAE (Mean Absolute Error)**

$$
L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

Characteristics: more robust to outliers, but gradient is constantly $\pm 1$ and non-differentiable at zero.

**Huber Loss (best of both)**

$$
L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}
$$

Uses MSE for small errors (smooth gradient) and MAE for large errors (capped penalty), combining the advantages of both.

| Loss | Outlier sensitivity | Gradient behavior | Use case |
|------|---------------------|-------------------|----------|
| MSE | Sensitive | Linear growth | Error distribution close to Gaussian |
| MAE | Robust | Constant ±1 | Outliers present |
| Huber | Balanced | Smooth transition | Need both advantages |

### 2.2 Classification Losses

**Cross-Entropy (CE)**

The standard loss for multi-class classification. True label $y$ is a one-hot vector, prediction $\hat{y}$ is the softmax output:

$$
L_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_c
$$

Where $c$ is the correct class. Intuition: the higher the probability assigned to the correct class, the closer loss is to 0; if extremely low probability is assigned to the correct class, loss tends to $+\infty$.

In PyTorch, `nn.CrossEntropyLoss()` accepts logits (not yet softmaxed) and internally applies `log_softmax + NLLLoss`, which is more stable than manual composition.

**Binary Cross-Entropy (BCE)**

$$
L_{\text{BCE}} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]
$$

In PyTorch, `nn.BCEWithLogitsLoss()` also accepts logits and uses a numerically stable sigmoid + log implementation internally.

**Focal Loss (Lin et al., 2017)**

Solves class imbalance. Standard cross-entropy gives loss near 0 for easy samples ($\hat{y}_c$ close to 1), but the accumulated loss from a large number of easy samples can still be substantial, drowning out the signal from hard samples.

$$
L_{\text{Focal}} = -(1 - \hat{y}_c)^\gamma \log \hat{y}_c
$$

$\gamma \geq 0$ is the focusing parameter:
- $\gamma = 0$: degenerates to standard cross-entropy
- $\gamma > 0$: easy samples ($\hat{y}_c \to 1$) have weight $(1 - \hat{y}_c)^\gamma \to 0$, automatically down-weighted
- $\gamma = 2$ is a common value

Intuition: make the model "focus" on samples it cannot classify well, rather than spending effort on those already correct.

> Key takeaway: the core of Focal Loss is the modulation factor $(1 - \hat{y}_c)^\gamma$ — the easier the sample, the lower its weight.

### 2.3 Contrastive Learning Losses

Contrastive learning does not rely on labels; instead it learns "similar things should be close, dissimilar things should be far apart."

**InfoNCE Loss**

$$
L = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\sum_{j=0}^{K} \exp(\text{sim}(q, k_j) / \tau)}
$$

Where $q$ is the query, $k_+$ is the positive sample, $k_j$ includes the positive sample and $K-1$ negative samples, $\tau$ is the temperature parameter, and $\text{sim}$ is a similarity function (usually cosine similarity).

Essentially: among the positive sample and all candidates, the softmax probability of the positive sample should be as high as possible.

**Symmetric cross-entropy (used by CLIP)**

CLIP performs contrastive learning on images and text, computing InfoNCE in both directions and averaging:

$$
L = \frac{1}{2}(L_{\text{image} \to \text{text}} + L_{\text{text} \to \text{image}})
$$

Each direction is just a softmax: given an image, the correct text should have the highest probability among all texts, and vice versa.

### 2.4 Label Smoothing

Standard classification targets are one-hot: correct class is 1, all others are 0. This encourages the model to output extreme probabilities (approaching 1 or 0), leading to overconfidence.

Label Smoothing softens hard labels:

$$
y_k^{\text{smooth}} = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K} & \text{if } k = c \\ \frac{\epsilon}{K} & \text{otherwise} \end{cases}
$$

$\epsilon$ is usually 0.1. Intuition: tell the model "you are only 90% sure about the correct class, and the remaining 10% is evenly distributed among the other classes." This prevents the model from outputting extreme logits and improves generalization.

> Label smoothing is almost standard in Transformer training; the original Transformer paper used $\epsilon = 0.1$.

---

## 3. Progressive Implementation

**Step 1 · Regression loss comparison**

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

print(f"Error=3: MSE={mse[-1]:.1f}, MAE={mae[-1]:.1f}, Huber={huber[-1]:.2f}")
print(f"Error=0.5: MSE={mse[58]:.4f}, MAE={mae[58]:.4f}")
# MSE penalizes large errors far more than MAE
```

**Step 2 · Cross-entropy and Focal Loss**

```python
import numpy as np

def cross_entropy(p_correct):
    """Standard cross-entropy (p_correct is the probability assigned to the correct class)"""
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
# At p_correct=0.99, Focal/CE is only 0.01%, easy samples are heavily suppressed
# At p_correct=0.1, Focal/CE is still 81%, hard samples are almost unaffected
```

**Step 3 · Label Smoothing implementation**

```python
import torch
import torch.nn as nn

def label_smoothing_manual(labels, num_classes, epsilon=0.1):
    """Manual label smoothing"""
    one_hot = torch.zeros(labels.size(0), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    smooth = one_hot * (1 - epsilon) + epsilon / num_classes
    return smooth

labels = torch.tensor([2])  # correct class is class 2
smooth_labels = label_smoothing_manual(labels, num_classes=5, epsilon=0.1)
print(f"Original one-hot: [0, 0, 1, 0, 0]")
print(f"After smoothing:  {smooth_labels[0].tolist()}")
# [0.02, 0.02, 0.92, 0.02, 0.02]

# Built-in PyTorch support
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
logits = torch.randn(1, 5)
target = torch.tensor([2])
loss = loss_fn(logits, target)
print(f"Label smoothing loss: {loss.item():.4f}")
```

**Step 4 · InfoNCE implementation and verification**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

BATCH, DIM = 4, 64
TEMPERATURE = 0.07

# Simulate queries and keys
q = F.normalize(torch.randn(BATCH, DIM), dim=-1)  # queries
k = F.normalize(torch.randn(BATCH, DIM), dim=-1)  # keys

# Similarity matrix: (batch, batch)
sim = q @ k.T / TEMPERATURE

# Diagonal entries are positive pairs (q_i and k_i), off-diagonal are negatives
labels = torch.arange(BATCH)

# One-direction InfoNCE = CrossEntropyLoss(sim, arange(batch))
loss = nn.CrossEntropyLoss()(sim, labels)
print(f"InfoNCE loss: {loss.item():.4f}")

# Symmetric version (CLIP-style): image→text + text→image
sim_t2i = k @ q.T / TEMPERATURE
loss_symmetric = (nn.CrossEntropyLoss()(sim, labels) + nn.CrossEntropyLoss()(sim_t2i, labels)) / 2
print(f"Symmetric InfoNCE loss: {loss_symmetric.item():.4f}")
```

---

## 4. Engineering Pitfalls (Sorted by Severity)

1. **log(0) causes NaN**
   Symptom: cross-entropy produces $\log(0) = -\infty$ when $\hat{y} = 0$.
   Fix: always use `nn.CrossEntropyLoss()` or `nn.BCEWithLogitsLoss()`; they have numerically stable implementations internally. Do not manually write `log(softmax(...))`.

2. **Cross-entropy dominated by majority class in imbalanced settings**
   Symptom: out of 100 samples, 95 are class A; the model gets 95% accuracy by always predicting A, and loss is also low.
   Fix: use Focal Loss to down-weight easy samples, oversample minority classes, or set class weights in the loss `nn.CrossEntropyLoss(weight=class_weights)`.

3. **Wrong temperature parameter in contrastive learning**
   Symptom: temperature $\tau$ too large → softmax becomes uniform, model learns no discrimination; too small → softmax becomes one-hot, gradients vanish.
   Fix: CLIP uses $\tau = 0.07$, SimCLR uses $\tau = 0.5$; tune within 0.05–0.5 depending on the task. $\tau$ can also be a learnable parameter.

4. **Label Smoothing epsilon set too large**
   Symptom: with $\epsilon = 0.4$ the model cannot learn clear class signals, and training loss stops decreasing.
   Fix: standard value is $\epsilon = 0.1$, generally not exceeding 0.2.

5. **Wrong loss for regression tasks**
   Symptom: a few extreme outliers exist in the data, and MSE gets pulled off-center by them.
   Fix: start with Huber Loss ($\delta=1.0$), or clean outliers first before deciding between MSE and MAE.

> Key takeaway: the key to choosing a loss function is "what do you want the model to focus on?" Regression cares about numerical accuracy, classification cares about class probabilities, and contrastive learning cares about relative distances.

---

## Evolution Notes

> **The evolution of loss functions**: from supervised learning (MSE → cross-entropy → Focal Loss) to contrastive learning (InfoNCE → symmetric cross-entropy), and on to RLHF (ranking loss for the Reward Model) — loss function design has always been answering the same question: "what signal will make the model learn the behavior we want?"
>
> The "modulation factor" idea of Focal Loss later also influenced hard example mining and curriculum learning.
>
> **New problems left behind**: contrastive losses need a large number of negative samples to learn well — this gave rise to MoCo's momentum encoder and SimCLR's large batch size strategy.

→ Next: [Backpropagation & Optimizers — How Do Models Learn from Mistakes?](../backpropagation/README_EN.md)

---

**Previous**: [Softmax & Probability Distributions](../softmax/README_EN.md) | **Next**: [Backpropagation & Optimizers](../backpropagation/README_EN.md)
