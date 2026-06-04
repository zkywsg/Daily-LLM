# 2015 · ResNet 与 Batch Norm：深度的解放

> **阶段**：视觉线
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

网络变深后训练误差反而升高，深度超过二十多层就容易退化。

## 发生了什么

ResNet 用跳跃连接学习残差，Batch Norm 稳定层输入分布，让极深网络可以训练。

## 解决了什么

网络深度被解放，152 层 ResNet 在 ImageNet 上达到低于人类水平的错误率。

## 留下了什么新问题

更深更宽的模型加剧算力需求，也让归一化和残差设计成为工程必修课。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[Residual Connection](../../foundations/structures/residual-connections/)** | 给梯度和信息提供直通路径，缓解退化问题。 |
| **[Batch Normalization](../../foundations/deep-learning/normalization/)** | 稳定中间激活分布，提高训练速度和稳定性。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [残差连接](../../foundations/structures/residual-connections/)
- [归一化](../../foundations/deep-learning/normalization/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [CNN 架构](../../tracks/vision/cnn-architectures/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
