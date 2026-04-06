# Phase 01 · 视觉线（2012–2017）

**[English](README_EN.md) | [中文](README.md)**

从手工特征到自动学习，CNN 如何一步步解放视觉理解的上限。
这条线在 2020 年 ViT 处与语言线汇流。

## 本阶段内容

### [训练与优化](training/README.md)
Dropout、Batch Norm、数据增强、GPU 训练技巧
- 正则化：Dropout、DropConnect（原理详见 [前置·正则化](../00-Prerequisites/regularization/README.md)）
- 归一化：Batch Norm、Layer Norm
- 优化器：SGD、Adam
- 训练稳定性工程

### [CNN 架构](cnn-architectures/README.md)
从”图像为什么不能直接交给全连接层”出发，沿着 AlexNet → ResNet 的问题链理解经典 CNN 演进，并收束到注意力出现前的局部建模边界。
- 卷积、感受野与下采样
- 深度、计算量与信息流动的权衡
- 经典 CNN 如何一步步逼近注意力时代

### [目标检测](object-detection/README.md)
从”分类一张图”到”找到图里所有东西”——检测范式从两阶段到单阶段的演进。
- R-CNN → Faster R-CNN 的区域提案革命
- YOLO → SSD 的单阶段检测与多尺度策略
- RetinaNet / Focal Loss 解决类别不平衡

### [分割与生成](segmentation-gan/README.md)
理解像素 vs 创造像素——编码器-解码器架构的分割与生成两副面孔。
- FCN / U-Net 的语义分割
- VAE 的变分生成与潜空间结构
- GAN / DCGAN / Progressive GAN 的生成对抗
- Neural Style Transfer 的内容与风格分离

### [GAN 进阶](gan-advanced/README.md)
从随机噪声到精确控制——条件 GAN、CycleGAN、Pix2Pix、StyleGAN 的可控生成之路。
- 条件 GAN：指定生成什么
- Pix2Pix / CycleGAN：配对与无配对图像翻译
- StyleGAN：分层风格控制与极致生成质量

### [轻量化架构](lightweight-vision/README.md)
大模型很好，但手机装不下——2016-2017 年的模型效率革命。
- SqueezeNet / MobileNet 的参数压缩策略
- SE-Net 的通道注意力
- 深度可分离卷积与 CNN 路线收尾

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2012 | AlexNet | 手工特征时代终结，深度学习元年 |
| 2013 | ZFNet / VAE | CNN 可视化；连续潜变量生成模型 |
| 2014 | VGGNet / GoogLeNet / GAN | 深度探索；多尺度架构；生成对抗训练 |
| 2015 | ResNet / Batch Norm | 152 层，Top-5 低于人类；训练速度飞跃 |
| 2016 | DenseNet / AlphaGo / WaveNet | 特征复用；CNN+RL 决策；自回归音频生成 |
| 2017 | SE-Net / Progressive GAN / CycleGAN | 通道注意力；渐进式高质量图像生成；无配对图像翻译 |
| 2018 | StyleGAN | 潜空间分层风格控制，面部生成达到照片级真实 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [语言线 →](../02-Language-Transformers/)
