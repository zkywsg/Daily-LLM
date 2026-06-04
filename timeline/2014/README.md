# 2014 · GAN、Seq2Seq、Attention、Adam：一年四响

> **阶段**：生成与序列
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

模型会分类却不擅长生成，机器翻译依赖规则对齐，优化学习率需要大量手工调节。

## 发生了什么

GAN 打开生成式建模，Seq2Seq 改写翻译范式，Attention 缓解固定向量瓶颈，Adam 简化优化。

## 解决了什么

生成、序列转换、动态对齐和自适应优化四个基础件同时到位。

## 留下了什么新问题

GAN 训练不稳定，Seq2Seq 仍串行，Attention 还没有成为统一架构。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[Attention](../../tracks/language/attention-mechanisms/)** | 让解码器按需回看输入位置，突破固定上下文瓶颈。 |
| **Adam** | 结合动量和自适应学习率，成为深度学习默认优化器。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [编码器-解码器](../../foundations/structures/encoder-decoder/)
- [注意力入门](../../foundations/structures/attention-primer/)
- [优化与调度](../../foundations/deep-learning/optimization-scheduling/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [注意力机制](../../tracks/language/attention-mechanisms/)
- [GAN 进阶](../../tracks/vision/gan-advanced/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
