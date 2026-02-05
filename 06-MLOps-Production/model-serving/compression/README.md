# Model Compression Strategies

**[English](README.md) | [中文](README_CN.md)**

## Table of Contents

1. [Background](#1-background)
2. [Core Concepts](#2-core-concepts)
3. [Mathematical Principles](#3-mathematical-principles)
4. [Code Implementation](#4-code-implementation)
5. [Experimental Comparison](#5-experimental-comparison)
6. [Best Practices and Common Pitfalls](#6-best-practices-and-common-pitfalls)
7. [Summary](#7-summary)

---

## 1. Background

### 1.1 Large Model Challenges

- **Deployment Cost**: Large models require expensive GPUs
- **Inference Latency**: Affects user experience
- **Power Consumption**: Edge devices cannot bear it
- **Storage**: Model files are too large

### 1.2 Value of Compression

- **Cost Reduction**: Save 70%+ computing resources
- **Speed Improvement**: 2-4x inference acceleration
- **Edge Deployment**: Run on mobile/IoT devices
- **Environmentally Friendly**: Reduce carbon emissions

---

## 2. Core Concepts

### 2.1 Quantization

Reduce parameter precision:
- FP32 → FP16: 2x compression
- FP32 → INT8: 4x compression
- FP32 → INT4: 8x compression

### 2.2 Pruning

Remove unimportant parameters:
- **Unstructured**: Set individual weights to zero
- **Structured**: Remove entire channels/layers

### 2.3 Distillation

Small model learns from large model knowledge.

---

## 3. Mathematical Principles

### 3.1 Compression Ratio

$$
\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}}
$$

### 3.2 Accuracy-Efficiency Trade-off

$$
\text{Efficiency} = \frac{\text{Accuracy}}{\text{Model Size}} \times \text{Speed}
$$

---

## 4. Code Implementation

### 4.1 INT8 Dynamic Quantization

```python
import torch

# Original model
model = MyModel().eval()

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Save
torch.save(quantized_model.state_dict(), "quantized.pth")
```

### 4.2 GPTQ 4-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Use directly
output = model.generate(input_ids, max_new_tokens=100)
```

### 4.3 Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    """Distillation loss"""

    def __init__(self, temperature=2.0):
        super().__init__()
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Soft target loss
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T * self.T)

        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, labels)

        return 0.7 * soft_loss + 0.3 * hard_loss

# Training loop
for batch in dataloader:
    inputs, labels = batch

    # Teacher model (frozen)
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)

    # Student model
    student_logits = student_model(inputs)

    # Distillation loss
    loss = distillation_loss(student_logits, teacher_logits, labels)

    loss.backward()
    optimizer.step()
```

---

## 5. Experimental Comparison

### 5.1 Method Comparison

| Method | Compression Ratio | Accuracy Loss | Speed Improvement | Use Case |
|--------|-----------------|---------------|------------------|-----------|
| **FP16** | 2x | <1% | 2x | General |
| **INT8** | 4x | 2-3% | 3x | General |
| **INT4** | 8x | 4-6% | 4x | Edge |
| **Pruning** | 2-10x | 3-5% | 2x | Specialized |
| **Distillation** | 10-100x | 5-10% | 10x | Specific |

### 5.2 Combination Effects

| Combination | Compression Ratio | Accuracy |
|------------|-----------------|----------|
| INT8 | 4x | 97% |
| INT4 | 8x | 94% |
| INT4 + Pruning | 16x | 92% |
| Distillation + INT8 | 40x | 90% |

---

## 6. Best Practices and Common Pitfalls

### 6.1 Best Practices

1. **Quantize first**: Try INT8 first, if not good then try INT4
2. **Calibration data**: Static quantization requires representative data
3. **Layer selection**: Some layers are sensitive to quantization, keep FP32
4. **Gradual compression**: Don't compress too much at once
5. **Comprehensive evaluation**: Focus on both accuracy and latency

### 6.2 Method Selection

```
Deployment environment?
├── Server GPU → FP16/INT8
├── Server CPU → INT8
├── Edge devices → INT4/Distillation
└── Mobile/IoT → Distillation + INT4
```

---

## 7. Summary

Model compression is essential for deploying large models:

1. **Quantization**: INT8 is general choice, INT4 for extreme compression
2. **Pruning**: Structured pruning is easier to deploy
3. **Distillation**: Compression during training, best effect but high cost
4. **Combination**: Use multiple methods together

**Recommended Strategies**:
- Server: INT8 quantization
- Edge: INT4 quantization
- Mobile: Distillation + INT4
- Cost-sensitive: GPTQ/AWQ
