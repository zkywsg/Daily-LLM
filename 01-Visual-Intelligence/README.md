# Phase 01 · 视觉线（2012–2017）

从手工特征到自动学习，CNN 如何一步步解放视觉理解的上限。
这条线在 2020 年 ViT 处与语言线汇流。

## 本阶段内容

### [CNN 架构](cnn-architectures/README.md)
AlexNet → ZFNet → VGGNet → GoogLeNet → ResNet → DenseNet
- 卷积、池化、感受野
- 深度 vs 宽度的系统探索
- 跳跃连接解决退化问题
- 轻量化架构：MobileNet、SqueezeNet、SE-Net

### [序列模型](sequence-models/README.md)
GAN、VAE、WaveNet，以及 AlphaGo（CNN + RL）
- 生成对抗网络的对抗训练原理
- 变分自编码器的潜变量空间
- 自回归波形生成
- 强化学习与 CNN 的结合

### [训练与优化](training/README.md)
Dropout、Batch Norm、数据增强、GPU 训练技巧
- 正则化：Dropout、DropConnect（原理详见 [前置·正则化](../00-Prerequisites/regularization/README.md)）
- 归一化：Batch Norm、Layer Norm
- 优化器：SGD、Adam
- 训练稳定性工程

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2012 | AlexNet | 手工特征时代终结，深度学习元年 |
| 2013 | ZFNet / VAE | CNN 可视化；连续潜变量生成模型 |
| 2014 | VGGNet / GoogLeNet / GAN | 深度探索；多尺度架构；生成对抗训练 |
| 2015 | ResNet / Batch Norm | 152 层，Top-5 低于人类；训练速度飞跃 |
| 2016 | DenseNet / AlphaGo / WaveNet | 特征复用；CNN+RL 决策；自回归音频生成 |
| 2017 | SE-Net / Progressive GAN | 通道注意力；渐进式高质量图像生成 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [语言线 →](../02-Language-Transformers/)
