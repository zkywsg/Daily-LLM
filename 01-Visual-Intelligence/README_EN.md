# Phase 01 · Vision Line (2012–2017)

**[English](README_EN.md) | [中文](README.md)**

From hand-crafted features to learned representations — how CNNs progressively raised the ceiling of visual understanding. This line converges with the Language Line at ViT in 2020.

## Modules

### [Training & Optimization](training/README_EN.md)
Dropout, Batch Normalization, data augmentation, GPU training techniques
- Regularization: Dropout, DropConnect (principles in [Prerequisites · Regularization](../00-Prerequisites/regularization/README_EN.md))
- Normalization: Batch Norm, Layer Norm
- Optimizers: SGD, Adam
- Training stability engineering

### [CNN Architectures](cnn-architectures/README_EN.md)
Starting from "why can't we just feed images into fully connected layers," follow the AlexNet → ResNet problem chain through classic CNN evolution, converging on the boundaries of local modeling before attention.
- Convolutions, receptive fields, and downsampling
- Trade-offs between depth, computation, and information flow
- How classic CNNs progressively approached the attention era

### [Object Detection](object-detection/README_EN.md)
From "classifying an image" to "finding everything in it" — the evolution from two-stage to one-stage detection paradigms.
- R-CNN → Faster R-CNN: the region proposal revolution
- YOLO → SSD: one-stage detection and multi-scale strategies
- RetinaNet / Focal Loss: solving class imbalance

### [Segmentation & Generation](segmentation-gan/README_EN.md)
Understanding pixels vs. creating pixels — two faces of the encoder-decoder architecture.
- FCN / U-Net for semantic segmentation
- GAN / DCGAN / Progressive GAN for generative adversarial training
- Neural Style Transfer: content and style disentanglement

### [Lightweight Architectures](lightweight-vision/README_EN.md)
Great models, but they don't fit on phones — the 2016–2017 efficiency revolution.
- SqueezeNet / MobileNet parameter compression strategies
- SE-Net channel attention
- Depthwise separable convolutions and the closing of the CNN roadmap

### [Sequence Models](sequence-models/README_EN.md)
RNN / LSTM / GRU / Seq2Seq main chapter has been migrated to [Language Line · Recurrent Networks & Seq2Seq](../02-Language-Transformers/recurrent-networks/README_EN.md)
- This page is retained for historical reference
- Still serves as a transitional entry point from the Vision Line

## Timeline Nodes

| Year | Work | Core Significance |
|------|------|-------------------|
| 2012 | AlexNet | End of hand-crafted features; deep learning year zero |
| 2013 | ZFNet / VAE | CNN visualization; continuous latent variable generative models |
| 2014 | VGGNet / GoogLeNet / GAN | Depth exploration; multi-scale architecture; adversarial training |
| 2015 | ResNet / Batch Norm | 152 layers, Top-5 below human; training speed breakthrough |
| 2016 | DenseNet / AlphaGo / WaveNet | Feature reuse; CNN+RL decision-making; autoregressive audio |
| 2017 | SE-Net / Progressive GAN | Channel attention; progressive high-quality image generation |

→ Full timeline: [00-Timeline](../00-Timeline/)

**Next Phase**: [Language Line →](../02-Language-Transformers/)
