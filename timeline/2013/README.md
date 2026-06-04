# 2013 · Word2Vec：词也能有坐标

> **阶段**：语言线
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

One-Hot 维度高且没有语义距离，模型不知道猫和狗比猫和飞机更接近。

## 发生了什么

Word2Vec 用上下文预测训练稠密词向量，让词之间的语义关系进入向量空间。

## 解决了什么

NLP 获得可迁移的语义表示，下游任务不再完全依赖稀疏人工特征。

## 留下了什么新问题

每个词仍只有一个静态向量，无法区分苹果公司和苹果水果。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **Skip-gram** | 用中心词预测上下文，适合学习稀有词表示。 |
| **VAE** | 用变分推断学习连续潜变量，推动生成模型发展。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [嵌入表示](../../foundations/representations/embeddings/)
- [概率与信息论](../../foundations/math/probability-information-theory/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [语言线总览](../../tracks/language/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
