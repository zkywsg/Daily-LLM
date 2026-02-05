# Attention Mechanisms

**[English](README.md) | [中文](README_CN.md)**

## Overview

Attention is the foundational mechanism that enables models to focus on relevant parts of input when producing output. It revolutionized sequence modeling and led to the Transformer architecture.

## Intuition

### Human Attention Analogy
When reading, you don't process every word equally—you focus on key parts relevant to your task.

**Example**:
```
Sentence: "The cat sat on the mat and looked at the bird"
Question: "Where is the cat?"
Attention: Focus on "sat on the mat"
```

## Core Concepts

### 1. Attention Score

Measure relevance between query and key:

```
Score(Q, K) = Q · K^T / √d_k
```

### 2. Attention Weights

Softmax normalized scores:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

| Component | Role | Analogy |
|-----------|------|---------|
| **Query** | What you're looking for | Question |
| **Key** | What might match | Index entries |
| **Value** | Actual information | Content |

## Types of Attention

### 1. Self-Attention

Each position attends to all positions in the same sequence.

```python
# Self-attention computation
scores = Q @ K.T / sqrt(dim)  # (seq, seq)
weights = softmax(scores)      # Attention distribution
output = weights @ V           # Weighted sum
```

**Use Cases**:
- Capture intra-sentence relationships
- Understand context within sequence
- Foundation of Transformers

### 2. Cross-Attention

Query from one sequence, Key/Value from another.

**Example**: Machine Translation
```
Source (English): "I love machine learning"
Target (French):   "J'adore l'apprentissage automatique"
                    ↓
French word "apprentissage" attends to "learning"
```

### 3. Multi-Head Attention

Parallel attention mechanisms focusing on different aspects.

```python
# Split into h heads
Q_split = split(Q, h)  # (batch, h, seq, d_k)
K_split = split(K, h)
V_split = split(V, h)

# Compute attention per head
head_outputs = [attention(q, k, v) for q, k, v in zip(Q_split, K_split, V_split)]

# Concatenate and project
output = concat(head_outputs) @ W_o
```

**Why Multiple Heads?**:
- Head 1: Syntactic relationships
- Head 2: Semantic relationships
- Head 3: Positional patterns
- ...

## Mathematical Formulation

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where:
- Q ∈ R^(n×d_k)  (queries)
- K ∈ R^(m×d_k)  (keys)
- V ∈ R^(m×d_v)  (values)
```

**Scaling Factor (√d_k)**:
Prevents softmax from entering regions with small gradients when d_k is large.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Attention | O(n²·d) | O(n²) |
| Linear | O(n·d²) | O(d²) |

## Implementation

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context), attention_weights

# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
output, weights = mha(x, x, x)  # Self-attention
print(f"Output shape: {output.shape}")  # (2, 100, 512)
print(f"Attention weights shape: {weights.shape}")  # (2, 8, 100, 100)
```

## Visualizing Attention

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, tokens):
    """
    attention_weights: (seq_len, seq_len)
    tokens: list of token strings
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='YlOrRd')
    plt.title('Attention Heatmap')
    plt.show()

# Example: Show which words attend to each other
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
plot_attention(weights[0, 0].detach().numpy(), tokens)
```

## Positional Encoding

Attention has no notion of position—positional encoding adds this information.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## From Attention to Transformers

**Key Insight**: Attention alone is sufficient—no need for recurrence or convolution.

**Benefits**:
1. **Parallelization**: Process all positions simultaneously
2. **Long-range dependencies**: Direct connection between any positions
3. **Interpretability**: Attention weights show what model focuses on

## Applications

### 1. Machine Translation
Cross-attention between source and target languages

### 2. Text Summarization
Self-attention identifies key sentences

### 3. Image Captioning
Cross-attention between image features and text

### 4. Speech Recognition
Attention aligns audio frames with text

## Best Practices

### 1. Masking
- **Padding Mask**: Ignore padded positions
- **Causal Mask**: Prevent attending to future (for autoregressive)

### 2. Initialization
- Xavier/Glorot for weight matrices
- Special care for residual connections

### 3. Optimization
- Gradient clipping for stability
- Learning rate warmup

## Common Pitfalls

1. **Not scaling**: Forgetting √d_k causes training instability
2. **Masking errors**: Incorrect masks lead to information leakage
3. **Memory issues**: O(n²) complexity problematic for long sequences

## Advanced Topics

### Sparse Attention
- **Local Attention**: Only attend to nearby positions
- **Strided Attention**: Attend to every k-th position
- **Linformer**: Linear complexity via low-rank approximation

### Efficient Transformers
- **Flash Attention**: Memory-efficient implementation
- **Multi-Query Attention**: Share K/V across heads

---

**Previous**: [Sequence Models](../../02-Neural-Networks/sequence-models/README.md) | **Next**: [Transformer Architecture](../transformer-architecture/README.md)
