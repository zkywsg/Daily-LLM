# 2024 · MoE、长上下文、o1：推理时慢思考

> **阶段**：系统生产
>
> 时间线主线在这一年发生了什么、解决了什么、又留下了什么新问题。

[← 回到主时间线](../README.md)

---

## 之前卡在哪

大模型推理成本随规模上涨，长上下文和复杂推理一步错就会连锁失败。

## 发生了什么

MoE 降低单次激活参数比例，长上下文扩展输入窗口，o1 类模型强调推理时计算。

## 解决了什么

能力提升不再只依赖训练期堆参数，推理期计算成为重要方向。

## 留下了什么新问题

路由、缓存、延迟、成本和评估都变成系统级问题。

---

## 同年关键工作

| 名字 | 贡献 |
|---|---|
| **[MoE](../../tracks/systems/model-serving/architecture/)** | 按 token 激活部分专家，在能力和成本之间折中。 |
| **Long Context** | 让模型处理更长文档和复杂任务上下文。 |

## 前置基础（Layer 0 · 工具箱）

> 看不懂当前内容时先来这里补课。

- [归纳偏置](../../foundations/structures/inductive-bias/)
- [数值精度](../../foundations/deep-learning/numerical-precision/)

## 主题深挖（Layer 2 · 主题深挖）

> 想沿这条主线纵向往后追，进入下面的 track。

- [模型服务架构](../../tracks/systems/model-serving/architecture/)
- [训练基础设施](../../tracks/systems/training-infrastructure/)

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
<!-- 不要手动编辑——改 timeline.ts 后重跑脚本 -->
