# Phase 02: 神经网络

> 从"特征要人设计"到"网络自己学"，再到"结构本身也可以学"。
> 这一阶段解决的核心问题是：什么样的归纳偏置，适合什么样的数据结构？

## 为什么这一阶段值得深入

Phase 01 建立了训练循环和评估方法的直觉。这一阶段进入具体的网络结构：

- **CNN** 为什么比全连接网络更适合图像？因为局部连接 + 权值共享 = 平移不变性
- **RNN / LSTM** 为什么适合序列？因为循环连接 = 隐式的时序记忆
- **ResNet** 为什么让网络可以很深？因为跳跃连接解决了梯度消失

读完这一阶段，你会理解：**架构设计不是魔法，而是对数据结构的归纳偏置**。

## 本阶段内容

### 1. [CNN 架构](cnn-architectures/README.md)
从 AlexNet 到 ResNet 的视觉架构演进。
- AlexNet / VGGNet / GoogLeNet / ResNet / DenseNet
- 轻量化架构：MobileNet、SqueezeNet、SE-Net
- 目标检测：Faster R-CNN、YOLO、SSD
- 卷积变体：深度可分离卷积、空洞卷积、1×1 卷积

### 2. [序列模型](sequence-models/README.md)
RNN、LSTM、GRU 与序列建模。
- 经典 RNN 及梯度消失问题
- LSTM 门控机制详解
- GRU：简化的门控方案
- 序列生成与采样策略

### 3. [训练与优化](training/README.md)
让网络真正收敛的工程技巧。
- Batch Normalization / Layer Normalization
- Dropout 与正则化变体
- 学习率调度策略
- 梯度裁剪与训练稳定性

## 时间线节点

本模块对应时间线中的以下关键工作：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2012 | AlexNet | 深度 CNN 的起点，证明特征可以自动学习 |
| 2012 | Dropout | 正则化标准手段，解决深度网络过拟合 |
| 2013 | VAE | 连续潜变量生成模型的数学基础 |
| 2013 | ZFNet | CNN 特征可视化，第一次看清网络在学什么 |
| 2014 | GAN | 生成对抗训练，生成式 AI 的第一块基石 |
| 2014 | VGGNet / GoogLeNet | 深度 vs 宽度的系统探索 |
| 2014 | GRU | 简化 LSTM，序列建模效率新选择 |
| 2015 | ResNet | 跳跃连接解放网络深度，视觉分类问题基本宣告解决 |
| 2015 | Batch Normalization | 训练速度提升数量级，允许更高学习率 |
| 2015 | DCGAN | 卷积 GAN 稳定训练，生成真实图像的第一个可复现方法 |
| 2016 | WaveNet | 自回归波形生成，序列生成质量飞跃 |
| 2016 | DenseNet | 特征复用最大化，参数效率极高 |
| 2017 | SE-Net | 通道注意力即插即用，ILSVRC 2017 冠军 |
| 2017 | Progressive GAN | 渐进式生成，高质量图像生成里程碑 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)
