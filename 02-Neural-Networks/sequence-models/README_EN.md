# Sequence Models

**[English](README_EN.md) | [中文](README.md)**

## Overview

Sequence models process sequential data like text, time series, and speech. This guide covers RNN, LSTM, GRU, and the transition to attention mechanisms.

## Recurrent Neural Networks (RNN)

### 1. Basic RNN

**Recurrent Connection**:
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

**Unrolled View**:
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T → Output
     ↑_________↑_________↑
```

### 2. Problems with Basic RNN

**Vanishing Gradients**:
- Gradients shrink exponentially through time steps
- Cannot capture long-term dependencies

**Exploding Gradients**:
- Gradients grow exponentially
- Cause unstable training

## Long Short-Term Memory (LSTM)

### Cell Structure

**Gates Control Information Flow**:

1. **Forget Gate**: What to discard
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```

2. **Input Gate**: What to store
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```

3. **Update Cell State**:
   ```
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
   ```

4. **Output Gate**: What to output
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(C_t)
   ```

### LSTM Variants

| Variant | Modification | Benefit |
|---------|-------------|---------|
| **Peephole** | Gates see cell state | Better timing |
| **Coupled** | Forget + input gates together | Fewer params |
| **BiLSTM** | Process both directions | Future context |

## Gated Recurrent Unit (GRU)

### Simplified Architecture

**Two Gates** (vs LSTM's three):

1. **Update Gate**: Balance old and new information
   ```
   z_t = σ(W_z · [h_{t-1}, x_t])
   ```

2. **Reset Gate**: How much past to forget
   ```
   r_t = σ(W_r · [h_{t-1}, x_t])
   ```

3. **Candidate Activation**:
   ```
   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
   ```

4. **Final Update**:
   ```
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
   ```

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 | 2 |
| Parameters | More | Fewer (25% less) |
| Performance | Similar | Similar |
| Speed | Slower | Faster |
| When to use | Long sequences | General purpose |

## Implementation

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(
    input_size=100,    # Embedding dimension
    hidden_size=128,   # Hidden state dimension
    num_layers=2,      # Stacked LSTMs
    batch_first=True,  # Input format: (batch, seq, feature)
    dropout=0.3,       # Dropout between layers
    bidirectional=True # Process both directions
)

# Forward pass
input_seq = torch.randn(32, 50, 100)  # (batch, seq_len, features)
output, (hidden, cell) = lstm(input_seq)
# output: (32, 50, 256) - concatenated forward+backward

# GRU (simpler, fewer params)
gru = nn.GRU(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

output, hidden = gru(input_seq)
```

## Applications

### 1. Text Classification
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]
        return self.fc(final_hidden)
```

### 2. Sequence to Sequence
Encoder-Decoder architecture for machine translation

### 3. Time Series Prediction
Financial forecasting, weather prediction

## Limitations & Transition

### Why RNNs Fall Short

1. **Sequential Processing**: Cannot parallelize
2. **Long Dependencies**: Information bottleneck
3. **Slow Training**: Step-by-step computation

### The Solution: Attention & Transformers

**Attention Mechanisms**:
- Direct connection between any positions
- No sequential dependency
- Parallel computation

See [Attention Mechanisms](../../03-NLP-Transformers/attention-mechanisms/README.md) for the evolution beyond RNNs.

## Best Practices

### 1. Handling Variable Lengths
```python
# Padding and Packing
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sort by length
lengths = [len(seq) for seq in sequences]
packed = pack_padded_sequence(padded_seqs, lengths, batch_first=True)
output, _ = lstm(packed)
output, _ = pad_packed_sequence(output, batch_first=True)
```

### 2. Regularization
- **Dropout**: Between layers (not on recurrent connections)
- **Weight Decay**: L2 regularization
- **Gradient Clipping**: Prevent exploding gradients

### 3. Initialization
- Xavier/Glorot for weights
- Orthogonal initialization for recurrent weights

## When to Use What

| Scenario | Recommendation |
|----------|---------------|
| **Short sequences (<50)** | GRU (faster) |
| **Long sequences (>100)** | LSTM (better memory) |
| **Bidirectional needed** | BiLSTM/BiGRU |
| **Maximum accuracy** | Transformer (see next) |
| **Resource constrained** | GRU or CNN |

---

**Previous**: [CNN Architectures](../cnn-architectures/README.md) | **Next**: [Training](../training/README.md)
