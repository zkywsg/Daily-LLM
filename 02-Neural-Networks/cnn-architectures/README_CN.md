[English](README.md) | [中文](README_CN.md)

# CNN 架构

## 概述

卷积神经网络（CNN）专用于处理网格状数据，尤其是图像。本指南涵盖了从 LeNet 到现代 EfficientNet 的经典 CNN 架构。

## 卷积层

### 1. 卷积运算

**二维卷积**：
```
Output[i,j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m,n]
```

**关键参数**：
- **卷积核大小（Kernel Size）**：通常为 3×3 或 5×5
- **步长（Stride）**：步长大小（通常为 1 或 2）
- **填充（Padding）**：保持空间维度
- **通道数（Channels）**：输入/输出特征图

### 2. 池化层

| 类型 | 操作 | 用途 |
|------|-----------|---------|
| **最大池化（Max Pooling）** | 取最大值 | 平移不变性 |
| **平均池化（Average Pooling）** | 取平均值 | 平滑特征 |
| **全局池化（Global Pooling）** | 降维至 1×1 | 最终特征提取 |

## 经典架构

### 1. LeNet (1998)
```
Input → Conv → Pool → Conv → Pool → FC → Output
```
- **首个成功的 CNN**
- 5 层，MNIST 手写数字识别

### 2. AlexNet (2012)
**赢得 ImageNet 竞赛的突破性架构**

```
Conv(11×11) → MaxPool → Conv(5×5) → MaxPool
→ Conv(3×3)×3 → MaxPool → FC(4096)×2 → FC(1000)
```

**创新点**：
- ReLU 激活函数
- GPU 训练
- Dropout 正则化
- 数据增强

### 3. VGGNet (2014)
**简洁性与深度**

| 变体 | 深度 | 配置 |
|---------|-------|---------------|
| VGG-16 | 16 | 13 个卷积层 + 3 个全连接层 |
| VGG-19 | 19 | 16 个卷积层 + 3 个全连接层 |

**核心思想**：堆叠 3×3 卷积 = 更大的感受野

```
64 → 64 → MaxPool → 128 → 128 → MaxPool
→ 256×2 → MaxPool → 512×2 → MaxPool → 512×2 → FC
```

### 4. ResNet (2015)
**残差连接解决梯度消失问题**

**残差块（Residual Block）**：
```
Output = F(x) + x
```

其中 F(x) 为残差映射（conv → BN → ReLU → conv）

| 变体 | 深度 | 参数量 | Top-1 准确率 |
|---------|-------|--------|-----------|
| ResNet-18 | 18 | 11.7M | 69.6% |
| ResNet-34 | 34 | 21.8M | 73.3% |
| ResNet-50 | 50 | 25.6M | 76.1% |
| ResNet-101 | 101 | 44.5M | 77.4% |

### 5. EfficientNet (2019)
**复合缩放：深度、宽度、分辨率**

**复合缩放公式**：
```
深度：d = α^φ
宽度：w = β^φ
分辨率：r = γ^φ

约束条件：α · β² · γ² ≈ 2
```

| 变体 | 参数量 | FLOPs | Top-1 准确率 |
|---------|--------|-------|-----------|
| B0 | 5.3M | 0.39B | 77.1% |
| B1 | 7.8M | 0.70B | 79.1% |
| B7 | 66M | 37B | 84.3% |

## 实现

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

# 使用预训练模型
from torchvision import models

# 加载预训练 ResNet-50
resnet = models.resnet50(pretrained=True)

# 针对新任务微调
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
```

## 现代技术

### 1. 批归一化（Batch Normalization）
```python
nn.BatchNorm2d(num_features)
```
归一化激活值：稳定训练，加速收敛

### 2. 跳跃连接（Skip Connections）
直接的梯度传播路径，防止梯度消失

### 3. CNN 中的注意力机制
- **Squeeze-and-Excitation**：通道注意力
- **CBAM**：通道 + 空间注意力

## 架构选择指南

| 使用场景 | 推荐架构 | 原因 |
|----------|-------------------------|--------|
| **移动端/边缘设备** | MobileNet, EfficientNet-Lite | 效率高 |
| **通用场景** | ResNet-50 | 平衡性好 |
| **高精度需求** | EfficientNet-B7, ResNet-152 | 性能最优 |
| **实时应用** | ResNet-18, MobileNet-V3 | 速度快 |
| **迁移学习** | ResNet, EfficientNet | 经过验证的特征 |

## 常见模式

### 1. 特征金字塔（Feature Pyramid）
用于目标检测的多尺度特征提取

### 2. 空洞卷积（Dilated Convolution）
在不损失分辨率的情况下增大感受野

### 3. 深度可分离卷积（Depthwise Separable Conv）
分解标准卷积以提高效率
```
深度卷积：每个通道应用单个滤波器
逐点卷积：1×1 卷积以组合通道
```

---

**上一篇**：[深度学习基础](../../01-Foundations/deep-learning-basics/README.md) | **下一篇**：[序列模型](../sequence-models/README.md)
