# 2017 · Transformer：把 RNN 扔掉

> **阶段**：语言线
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

LSTM 天生串行，长句训练慢，远距离依赖仍然难以稳定建模。

## 发生了什么

Transformer 完全用自注意力和前馈网络替代循环结构，让序列建模可以大规模并行。

## 解决了什么

它统一了上下文建模方式，成为后续 BERT、GPT 和大语言模型的基础架构。

## 留下了什么新问题

自注意力带来 O(n²) 复杂度，长上下文成本开始成为核心瓶颈。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[Self-Attention](../../tracks/language/attention-mechanisms/)** | 每个 token 直接和其他 token 建立依赖。 |
| **[Multi-Head Attention](../../tracks/language/transformer-architecture/)** | 在多个子空间并行建模不同关系。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [注意力入门](../../foundations/structures/attention-primer/)
- [Softmax](../../foundations/representations/softmax/)
- [归一化](../../foundations/deep-learning/normalization/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [Transformer 架构](../../tracks/language/transformer-architecture/)
- [注意力机制](../../tracks/language/attention-mechanisms/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
