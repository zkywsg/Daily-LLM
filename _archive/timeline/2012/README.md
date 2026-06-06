# 2012 · AlexNet：一声炮响，旧世界终结

> **阶段**：视觉线
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

计算机视觉依赖 SIFT、HOG 等手工特征，ImageNet 错误率多年停在 25% 到 26%。

## 发生了什么

AlexNet 用深度卷积网络、ReLU、GPU 并行、Dropout 和数据增强，把 Top-5 错误率打到 15.3%。

## 解决了什么

它证明特征可以从数据中学习，深度学习成为视觉识别的主线。

## 留下了什么新问题

大数据、GPU 算力和模型可解释性成为新的门槛。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[Dropout](../../foundations/deep-learning/regularization/)** | 随机失活缓解过拟合，成为深度网络标准正则化手段。 |
| **[ReLU](../../foundations/deep-learning/activation-functions/)** | 改善梯度流动，让深层网络训练明显加速。 |
| **GPU 训练** | 确立 CUDA 加速深度网络训练的基础设施范式。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [反向传播](../../foundations/deep-learning/backpropagation/)
- [激活函数](../../foundations/deep-learning/activation-functions/)
- [正则化](../../foundations/deep-learning/regularization/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [CNN 架构](../../tracks/vision/cnn-architectures/)
- [训练基础](../../tracks/vision/training/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
