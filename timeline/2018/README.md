# 2018 · BERT 与 GPT-1：预训练时代

> **阶段**：预训练
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

静态词向量无法处理一词多义，下游任务通常需要从头训练或重做大量特征工程。

## 发生了什么

BERT 用双向 Masked LM 学上下文表示，GPT-1 用自回归预训练展示生成式迁移。

## 解决了什么

预训练加微调成为 NLP 标准范式，词义开始随上下文动态变化。

## 留下了什么新问题

预训练成本上升，模型路线分化为理解式编码器和生成式解码器。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[BERT](../../tracks/language/pretrained-models/)** | 双向上下文预训练刷新多项 NLP 理解任务。 |
| **GPT-1** | 证明自回归语言模型可以迁移到下游任务。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [Tokenization](../../foundations/representations/tokenization/)
- [嵌入表示](../../foundations/representations/embeddings/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [预训练模型](../../tracks/language/pretrained-models/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
