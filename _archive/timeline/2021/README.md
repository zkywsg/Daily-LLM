# 2021 · CLIP、Codex、LoRA：多模态与效率

> **阶段**：多模态
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

视觉和语言系统割裂，大模型微调需要复制全部参数，成本只有巨头承担得起。

## 发生了什么

CLIP 用图文对比学习对齐视觉语言，Codex 把语言模型迁移到代码，LoRA 降低微调成本。

## 解决了什么

多模态对齐、代码生成和参数高效微调同时进入实用阶段。

## 留下了什么新问题

数据版权、模型偏见和微调后的部署治理成为新问题。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[CLIP](../../tracks/scale-multimodal/multimodal/)** | 用自然语言监督视觉模型，连接图像和文本空间。 |
| **[LoRA](../../tracks/alignment/peft/)** | 通过低秩适配器降低大模型微调成本。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [嵌入表示](../../foundations/representations/embeddings/)
- [损失函数](../../foundations/deep-learning/loss-functions/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [多模态](../../tracks/scale-multimodal/multimodal/)
- [PEFT](../../tracks/alignment/peft/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
