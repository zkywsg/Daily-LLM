[English](README_EN.md) | [中文](README.md)

# Phase 00 · Prerequisites

The minimum neural-network foundations required before entering the visual track and the language track.
Does not cover classical ML algorithms (SVM, decision trees, K-Means, etc.).

## Contents

### Foundations

| # | Module | Core Content |
|---|--------|--------------|
| 0 | [Probability & Information Theory](probability-information-theory/README_EN.md) | Probability distributions, Bayes' theorem, entropy, cross-entropy, KL divergence, MLE, mutual information |
| 1 | [Deep Learning Basics](deep-learning-basics/README_EN.md) | Neurons & forward propagation, backpropagation & gradient descent, loss functions & the training loop |
| 2 | [Linear Algebra](linear-algebra/README_EN.md) | Vectors & matrices, matrix multiplication as linear transformation, norms, SVD, broadcasting |
| 3 | [Softmax & Probability Distributions](softmax/README_EN.md) | Softmax formula, relationship with Sigmoid, temperature parameter, log-sum-exp trick |
| 4 | [Loss Functions](loss-functions/README_EN.md) | Regression/MSE/Huber, cross-entropy/Focal Loss, contrastive losses/InfoNCE, Label Smoothing |
| 5 | [Backpropagation & Optimizers](backpropagation/README_EN.md) | Chain rule & computation graphs, vanishing/exploding gradients, SGD/Momentum/Adam |
| 6 | [Learning-Rate Scheduling & Gradient Control](optimization-scheduling/README_EN.md) | Warmup/Cosine Decay/Step Decay, gradient clipping, AdamW/LAMB |

### Architecture Components

| # | Module | Core Content |
|---|--------|--------------|
| 7 | [Normalization](normalization/README_EN.md) | BatchNorm (train vs eval), LayerNorm, Pre-LN vs Post-LN |
| 8 | [Residual Connections](residual-connections/README_EN.md) | Gradient highways, projection shortcuts, Pre-LN vs Post-LN, DenseNet |
| 9 | [Activation Functions](activation-functions/README_EN.md) | Sigmoid's vanishing gradients & ReLU's rise, Dying ReLU & variants, GELU |
| 10 | [Regularization & Dropout](regularization/README_EN.md) | Overfitting diagnosis, Dropout variants, L2 weight decay, Early Stopping |

### NLP Bridge

| # | Module | Core Content |
|---|--------|--------------|
| 11 | [Embeddings](embeddings/README_EN.md) | One-hot → Embedding lookup, Word2Vec Skip-gram, static vs contextual embeddings |
| 12 | [Tokenization](tokenization/README_EN.md) | BPE/WordPiece/Unigram algorithms, vocab-size tradeoffs, special tokens |
| 13 | [Encoder-Decoder Paradigm](encoder-decoder/README_EN.md) | Seq2Seq, three Transformer paradigms (encoder-only/decoder-only/encoder-decoder) |
| 14 | [Attention Primer](attention-primer/README_EN.md) | Seq2Seq bottleneck, QKV framework, Scaled Dot-Product Attention, causal mask |

### Conceptual Bridge

| # | Module | Core Content |
|---|--------|--------------|
| 15 | [Inductive Bias](inductive-bias/README_EN.md) | Inductive-bias comparison of CNN/RNN/Transformer, ViT's data-bias tradeoff |
| 16 | [Numerical Precision & Distributed Training](numerical-precision/README_EN.md) | FP32/FP16/BF16, mixed-precision training, data parallelism, gradient accumulation |

## Suggested Reading Order

```
Foundations: 0 → 1 → 2 → 3 → 4 → 5 → 6
Architecture: 7 → 8 → 9 → 10
NLP Bridge: 11 → 12 → 13 → 14
Conceptual Bridge: 15 → 16
```

Each module follows the same structure: **Where it comes from → Intuition → Mechanics → Progressive Implementation → Engineering Pitfalls → Evolution Notes**.

## Timeline Nodes

| Year | Work | Core Significance |
|------|------|-------------------|
| 1812 | Bayes' theorem (Laplace) | Mathematical foundation of inverse-probability reasoning |
| 1948 | Shannon's information theory | Mathematical foundation of entropy and mutual information |
| 1986 | Backpropagation | Foundation for training multi-layer networks |
| 2010 | ReLU (Nair & Hinton) | Replaced sigmoid, restored gradient flow |
| 2012 | GPU deep-learning ecosystem | CUDA-accelerated training, compute infrastructure established |
| 2013 | Word2Vec | Word embeddings moved from sparse to dense |
| 2014 | Adam optimizer | The default optimizer that barely needs LR tuning |
| 2015 | Batch Normalization | Training speed improved by orders of magnitude, allowed higher learning rates |
| 2015 | ResNet (residual connections) | Made ultra-deep networks (152 layers) trainable |
| 2015 | Bahdanau Attention | Solved the fixed-length bottleneck of Seq2Seq |
| 2016 | BPE tokenization | Subword tokenization became the NLP standard |
| 2017 | AdamW | Decoupled weight decay, de-facto standard for Transformer training |
| 2017 | One-Cycle Policy | Single-cycle LR scheduling, improved training efficiency |
| 2017 | Focal Loss | Solved class imbalance, standard for object detection |
| 2018 | Mixed Precision | FP16 training doubled speed and halved memory |
| 2019 | LAMB | Large-batch optimizer supporting batch sizes 8K+ |

→ Full timeline at [../timeline](../timeline/)

**Next Phase**: [Visual Intelligence →](../tracks/vision/)
