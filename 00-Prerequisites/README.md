# Phase 00 · 前置准备

进入视觉线和语言线之前所需的最低限度神经网络基础。
不包含经典机器学习算法（SVM、决策树、K-Means 等）。

## 本阶段内容

### 基础概念

| # | 模块 | 核心内容 |
|---|------|---------|
| 0 | [概率与信息论基础](probability-information-theory/README.md) | 概率分布、贝叶斯定理、信息熵、交叉熵与 KL 散度、最大似然估计、互信息 |
| 1 | [深度学习基础](deep-learning-basics/README.md) | 神经元与前向传播、反向传播与梯度下降、损失函数与训练循环 |
| 2 | [线性代数基础](linear-algebra/README.md) | 向量与矩阵、矩阵乘法 = 线性变换、范数、SVD、广播机制 |
| 3 | [Softmax 与概率分布](softmax/README.md) | Softmax 公式、与 Sigmoid 的关系、温度参数、log-sum-exp trick |
| 4 | [损失函数全景](loss-functions/README.md) | 回归/MSE/Huber、交叉熵/Focal Loss、对比损失/InfoNCE、Label Smoothing |
| 5 | [反向传播与优化器](backpropagation/README.md) | 链式法则与计算图推导、梯度消失/爆炸与对策、SGD/Momentum/Adam |
| 6 | [学习率调度与梯度优化](optimization-scheduling/README.md) | Warmup/Cosine Decay/Step Decay、梯度裁剪、AdamW/LAMB |

### 架构组件

| # | 模块 | 核心内容 |
|---|------|---------|
| 7 | [激活函数家族](activation-functions/README.md) | Sigmoid 的梯度消失与 ReLU 的崛起、Dying ReLU 与变体、GELU |
| 8 | [归一化机制](normalization/README.md) | BatchNorm（训练/推理差异）、LayerNorm、Pre-LN vs Post-LN |
| 9 | [残差连接](residual-connections/README.md) | 梯度直通通道、Projection Shortcut、Pre-LN vs Post-LN、DenseNet |
| 10 | [正则化与 Dropout](regularization/README.md) | 过拟合诊断、Dropout 变体、L2 权重衰减、Early Stopping |

### NLP 桥梁

| # | 模块 | 核心内容 |
|---|------|---------|
| 11 | [Embedding 向量](embeddings/README.md) | One-hot → Embedding lookup、Word2Vec Skip-gram、静态 vs 上下文 embedding |
| 12 | [分词器](tokenization/README.md) | BPE/WordPiece/Unigram 算法、词表大小权衡、Special tokens |
| 13 | [编码器-解码器范式](encoder-decoder/README.md) | Seq2Seq、三种 Transformer 范式（encoder-only/decoder-only/encoder-decoder） |
| 14 | [注意力机制动机](attention-primer/README.md) | Seq2Seq 瓶颈、QKV 框架、Scaled Dot-Product Attention、因果 mask |

### 概念桥梁

| # | 模块 | 核心内容 |
|---|------|---------|
| 15 | [归纳偏置](inductive-bias/README.md) | CNN/RNN/Transformer 的归纳偏置对比、ViT 的数据-偏置权衡 |
| 16 | [数值精度与分布式训练](numerical-precision/README.md) | FP32/FP16/BF16、混合精度训练、数据并行、梯度累积 |

## 建议阅读顺序

```
基础概念: 0 → 1 → 2 → 3 → 4 → 5 → 6
架构组件: 7 → 8 → 9 → 10
NLP 桥梁: 11 → 12 → 13 → 14
概念桥梁: 15 → 16
```

每个模块内部遵循统一格式：**问题从哪来 → 直觉 → 机制 → 渐进式实现 → 工程陷阱 → 演进笔记**。

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 1812 | Bayes 定理（Laplace） | 逆概率推理的数学基础 |
| 1948 | Shannon 信息论 | 信息熵、互信息的数学基础 |
| 1986 | 反向传播 | 多层网络可训练的基础 |
| 2010 | ReLU (Nair & Hinton) | 取代 sigmoid，梯度流动恢复 |
| 2012 | GPU 深度学习生态 | CUDA 加速训练，计算基础设施确立 |
| 2013 | Word2Vec | 词嵌入从稀疏走向稠密 |
| 2014 | Adam 优化器 | 几乎不需要调学习率的默认优化器 |
| 2015 | Batch Normalization | 训练速度提升数量级，允许更高学习率 |
| 2015 | ResNet (残差连接) | 让超深网络（152层）可训练 |
| 2015 | Bahdanau Attention | 解决 Seq2Seq 的固定长度瓶颈 |
| 2016 | BPE 分词 | 子词级分词成为 NLP 标准 |
| 2017 | AdamW | 权重衰减解耦，Transformer 训练事实标准 |
| 2017 | One-Cycle Policy | 单周期学习率调度，训练效率提升 |
| 2017 | Focal Loss | 解决类别不平衡，目标检测标准 |
| 2018 | Mixed Precision | FP16 训练速度翻倍，显存减半 |
| 2019 | LAMB | 大 batch 训练优化器，支持 batch size 8K+ |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [视觉线 →](../01-Visual-Intelligence/)
