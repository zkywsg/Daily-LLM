# Transformer Architecture

**[English](README_EN.md) | [中文](README.md)**

## Overview

The Transformer is a revolutionary architecture introduced in "Attention Is All You Need" (2017) that replaced recurrence with attention, enabling parallelization and superior performance on sequence tasks.

## Architecture

### High-Level Structure

```
Input Embedding + Positional Encoding
    ↓
[Encoder Stack × N]
    ↓
[Decoder Stack × N] ← (for seq2seq)
    ↓
Output Projection + Softmax
```

### Encoder-Decoder Structure

**Encoder**: Processes input sequence
- Multi-Head Self-Attention
- Feed-Forward Network
- Layer Normalization & Residuals

**Decoder**: Generates output sequence
- Masked Multi-Head Self-Attention
- Cross-Attention (to encoder)
- Feed-Forward Network

## Core Components

### 1. Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, mask)[0]
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 2. Decoder Layer

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn1 = self.masked_self_attn(x, x, x, tgt_mask)[0]
        x = self.norm1(x + self.dropout(attn1))
        
        # Cross-attention to encoder
        attn2 = self.cross_attn(x, encoder_output, encoder_output, src_mask)[0]
        x = self.norm2(x + self.dropout(attn2))
        
        # Feed-forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        
        return x
```

## Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_encoder_layers=6, num_decoder_layers=6, 
                 d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and add positional encoding
        src = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        # Encode
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # Decode
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(tgt)
        return output
```

## Key Design Decisions

### 1. Layer Normalization vs Batch Normalization

**LayerNorm**: Normalizes across features
- More stable for sequences
- Independent of batch size

```python
# LayerNorm
mean = x.mean(-1, keepdim=True)
std = x.std(-1, keepdim=True)
output = (x - mean) / (std + eps) * gamma + beta
```

### 2. Residual Connections

Enable gradient flow through deep networks:
```
output = LayerNorm(x + Sublayer(x))
```

### 3. Position-wise FFN

Two linear transformations with ReLU:
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

## Variants

### 1. Encoder-Only (BERT-style)

**Use Cases**: Classification, tagging, representation

```python
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, 12, 3072)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        # Use [CLS] token or mean pooling
        return self.pooler(x[:, 0])
```

### 2. Decoder-Only (GPT-style)

**Use Cases**: Text generation, language modeling

```python
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Masked self-attention layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, 12, 3072)  # Same as encoder but with causal mask
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        # Causal mask
        mask = torch.tril(torch.ones(x.size(1), x.size(1)))
        
        x = self.pos_encoding(self.token_embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(x)
```

## Hyperparameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| d_model | 512-2048 | Model dimension |
| num_heads | 8-16 | Attention heads |
| num_layers | 6-24 | Encoder/decoder layers |
| d_ff | 2048-8192 | FFN hidden dimension |
| dropout | 0.1 | Regularization |

## Training Tips

### 1. Learning Rate Schedule

**Warmup + Decay**:
```python
def lr_schedule(step, warmup_steps, d_model):
    if step < warmup_steps:
        return step / warmup_steps
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
```

### 2. Label Smoothing

Prevent overconfidence:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 3. Gradient Accumulation

Simulate large batch sizes:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Efficiency Improvements

### 1. KV Cache

For autoregressive generation:
```python
# Cache previous keys and values
past_key_values = None
for token in sequence:
    output, past_key_values = model(token, past_key_values)
```

### 2. Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Flash Attention

Memory-efficient attention implementation

## Applications

| Task | Architecture | Example |
|------|-------------|---------|
| Translation | Encoder-Decoder | Original Transformer |
| Classification | Encoder-only | BERT |
| Generation | Decoder-only | GPT |
| Summarization | Encoder-Decoder | BART |
| QA | Encoder-only | RoBERTa |

## Common Pitfalls

1. **Wrong masking**: Causal mask for decoder
2. **Scaling**: Forget √d_k in attention
3. **Position encoding**: Not adding before first layer
4. **LayerNorm placement**: Pre-LN vs Post-LN

---

**Previous**: [Attention Mechanisms](../attention-mechanisms/README.md) | **Next**: [Pre-trained Models](../pretrained-models/README.md)
