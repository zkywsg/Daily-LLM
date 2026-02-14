# CNN Architectures

**[English](README_EN.md) | [中文](README.md)**

## Overview

Convolutional Neural Networks (CNNs) are specialized for processing grid-like data, particularly images. This guide covers classic CNN architectures from LeNet to modern EfficientNet.

## Convolutional Layers

### 1. Convolution Operation

**2D Convolution**:
```
Output[i,j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m,n]
```

**Key Parameters**:
- **Kernel Size**: Typically 3×3 or 5×5
- **Stride**: Step size (usually 1 or 2)
- **Padding**: Preserve spatial dimensions
- **Channels**: Input/output feature maps

### 2. Pooling Layers

| Type | Operation | Purpose |
|------|-----------|---------|
| **Max Pooling** | Take maximum value | Translation invariance |
| **Average Pooling** | Take average | Smooth features |
| **Global Pooling** | Reduce to 1×1 | Final feature extraction |

## Classic Architectures

### 1. LeNet (1998)
```
Input → Conv → Pool → Conv → Pool → FC → Output
```
- **First successful CNN**
- 5 layers, MNIST handwritten digits

### 2. AlexNet (2012)
**Breakthrough architecture that won ImageNet**

```
Conv(11×11) → MaxPool → Conv(5×5) → MaxPool
→ Conv(3×3)×3 → MaxPool → FC(4096)×2 → FC(1000)
```

**Innovations**:
- ReLU activation
- GPU training
- Dropout regularization
- Data augmentation

### 3. VGGNet (2014)
**Simplicity and depth**

| Variant | Depth | Configuration |
|---------|-------|---------------|
| VGG-16 | 16 | 13 conv + 3 FC |
| VGG-19 | 19 | 16 conv + 3 FC |

**Key Insight**: 3×3 convolutions stacked = larger receptive field

```
64 → 64 → MaxPool → 128 → 128 → MaxPool
→ 256×2 → MaxPool → 512×2 → MaxPool → 512×2 → FC
```

### 4. ResNet (2015)
**Residual connections solve vanishing gradients**

**Residual Block**:
```
Output = F(x) + x
```

Where F(x) is the residual mapping (conv → BN → ReLU → conv)

| Variant | Depth | Params | Top-1 Acc |
|---------|-------|--------|-----------|
| ResNet-18 | 18 | 11.7M | 69.6% |
| ResNet-34 | 34 | 21.8M | 73.3% |
| ResNet-50 | 50 | 25.6M | 76.1% |
| ResNet-101 | 101 | 44.5M | 77.4% |

### 5. EfficientNet (2019)
**Compound scaling: depth, width, resolution**

**Compound Scaling Formula**:
```
depth: d = α^φ
width: w = β^φ
resolution: r = γ^φ

s.t. α · β² · γ² ≈ 2
```

| Variant | Params | FLOPs | Top-1 Acc |
|---------|--------|-------|-----------|
| B0 | 5.3M | 0.39B | 77.1% |
| B1 | 7.8M | 0.70B | 79.1% |
| B7 | 66M | 37B | 84.3% |

## Implementation

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# Usage with pretrained models
from torchvision import models

# Load pretrained ResNet-50
resnet = models.resnet50(pretrained=True)

# Fine-tune for new task
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
```

## Modern Techniques

### 1. Batch Normalization
```python
nn.BatchNorm2d(num_features)
```
Normalizes activations: stabilizes training, faster convergence

### 2. Skip Connections
Direct gradient flow paths prevent vanishing gradients

### 3. Attention in CNNs
- **Squeeze-and-Excitation**: Channel attention
- **CBAM**: Channel + Spatial attention

## Architecture Selection Guide

| Use Case | Recommended Architecture | Reason |
|----------|-------------------------|--------|
| **Mobile/Edge** | MobileNet, EfficientNet-Lite | Efficiency |
| **General Purpose** | ResNet-50 | Balanced |
| **High Accuracy** | EfficientNet-B7, ResNet-152 | Performance |
| **Real-time** | ResNet-18, MobileNet-V3 | Speed |
| **Transfer Learning** | ResNet, EfficientNet | Proven features |

## Common Patterns

### 1. Feature Pyramid
Multi-scale feature extraction for object detection

### 2. Dilated Convolution
Increase receptive field without losing resolution

### 3. Depthwise Separable Conv
Factorize standard conv for efficiency
```
Depthwise: Apply single filter per channel
Pointwise: 1×1 conv to combine channels
```

---

**Previous**: [Deep Learning Basics](../../01-Foundations/deep-learning-basics/README.md) | **Next**: [Sequence Models](../sequence-models/README.md)
