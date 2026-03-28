[English](README_EN.md) | [中文](README.md)

# CNN 架构

## 概述

卷积神经网络（CNN）是图像任务的基础范式。与全连接网络相比，CNN 利用局部连接与参数共享，把“像素空间”逐步变成“语义空间”。本章面向已有机器学习基础的读者，重点是看懂 CNN 的设计动机与架构演进逻辑，而不仅是记住模型名字。

## 学习目标

完成本章后，你应能回答：

1. 卷积层、池化层分别在解决什么问题？
2. LeNet -> AlexNet -> VGG -> ResNet -> EfficientNet 的演进主线是什么？
3. 不同资源约束下如何做 CNN 架构选型？

## 1. 卷积到底在做什么

二维卷积可写为：

$$
Y(i,j)=\sum_m\sum_n X(i+m,j+n)\cdot K(m,n)
$$

直觉上，卷积核就是一个可学习的“局部模式探测器”：某些核对边缘敏感，某些核对纹理敏感。网络越深，探测到的模式越抽象。

关键超参数：

- Kernel Size：决定局部感受野，常见 `3x3`
- Stride：控制下采样速度
- Padding：控制边界信息保留
- Channels：控制特征表达容量

你要记住：卷积层的核心价值是“以更少参数提取空间结构信息”。

## 2. 池化与下采样

池化的目标是降低分辨率并提升鲁棒性。

| 类型 | 操作 | 常见作用 |
|------|------|----------|
| Max Pooling | 取局部最大值 | 保留显著响应，增强平移鲁棒性 |
| Average Pooling | 取局部平均值 | 平滑特征，降低噪声 |
| Global Average Pooling | 每通道压到 `1x1` | 取代大 FC，减少参数量 |

你要记住：下采样不只是降计算量，也是在做“信息压缩与不变性建模”。

## 3. 架构演进主线

### 3.1 LeNet（1998）

`Input -> Conv -> Pool -> Conv -> Pool -> FC -> Output`

开创了“卷积 + 池化”的图像识别范式，适用于小规模灰度图任务。

### 3.2 AlexNet（2012）

主线贡献：更深网络 + ReLU + Dropout + 数据增强 + GPU 训练。  
它证明了“数据规模 + 算力 + 更深模型”的组合能够显著提升性能。

### 3.3 VGG（2014）

主线贡献：统一使用小卷积核（`3x3`）堆叠，通过深度提升表达力。  
优点是结构规整，缺点是参数和计算开销较高。

### 3.4 ResNet（2015）

残差块核心：

$$
y = F(x) + x
$$

直接给梯度提供“短路径”，显著缓解深层网络退化问题。

### 3.5 EfficientNet（2019）

核心思想：不是只放大深度或宽度，而是联合缩放三者：

$$
d=\alpha^\phi,\quad w=\beta^\phi,\quad r=\gamma^\phi,\quad \alpha\cdot\beta^2\cdot\gamma^2\approx2
$$

在精度与效率之间取得更优平衡。

你要记住：CNN 演进本质在优化三件事: 表达能力、可训练性、计算效率。

## 4. ResNet 残差块实现（PyTorch）

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + identity)
        return out
```

## 5. 迁移学习最小模板

```python
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

说明：

- 小数据集优先迁移学习，而不是从零训练
- 先训分类头，再逐步解冻 backbone 往往更稳

你要记住：工程上最常用 CNN 方案不是“造新网络”，而是“基于预训练模型微调”。

## 6. 现代 CNN 常用模块

1. BatchNorm：稳定特征分布，加速收敛
2. Skip Connection：改善梯度传播
3. SE/CBAM 注意力：按通道或空间重加权特征
4. Depthwise Separable Conv：显著降 FLOPs，移动端常用
5. Dilated Conv：不降分辨率时扩大感受野

## 7. 架构选型指南

| 场景 | 推荐起点 | 选型理由 |
|------|----------|----------|
| 通用分类基线 | ResNet-50 | 社区成熟、微调稳定 |
| 移动端部署 | MobileNetV3 / EfficientNet-Lite | 延迟和功耗更优 |
| 高精度离线任务 | EfficientNet-Bx / ConvNeXt 大模型 | 精度上限高 |
| 实时系统 | ResNet-18 / MobileNet 小模型 | 吞吐更高 |
| 小样本迁移 | 预训练 ResNet/EfficientNet | 收敛快、泛化更稳 |

## 8. 排障与调参优先级

当效果不达标，按顺序检查：

1. 输入分辨率是否与任务匹配（过低会丢关键信息）
2. 数据增强是否过强或过弱
3. 学习率与 warmup 是否合理
4. 模型容量是否不足（欠拟合）或过大（过拟合）
5. 推理时预处理是否与训练一致

## 9. 常见误区

- 误区 1：盲目追求更深网络  
  正解：先确认数据规模、计算预算与推理延迟约束。
- 误区 2：只看 Top-1 精度  
  正解：同时看吞吐、延迟、显存、可部署性。
- 误区 3：忽略数据分布偏移  
  正解：上线前做目标场景验证集评估。

## 下一步学习

- 继续时序建模：进入 [序列模型](../sequence-models/README.md)
- 继续注意力范式：进入 [Transformer 架构](../../03-NLP-Transformers/transformer-architecture/README.md)

---

**上一篇**：[深度学习基础](../../01-Foundations/deep-learning-basics/README.md) | **下一篇**：[序列模型](../sequence-models/README.md)
