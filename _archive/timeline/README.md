# 深度学习与大模型演进时间线

> 每一个技术的出现，背后都有一个"不得不解决"的问题。
> 这条时间线不是论文列表，而是一部"被逼出来的历史"。

<img src="assets/timeline.svg" alt="深度学习与大模型演进地图" width="100%">

主时间线从 2012 AlexNet 起。1948 / 1958 / 1986 / 1997 四个里程碑作为
「深度学习前史」单列，详见 [prehistory/](./prehistory/)。

## 导航

| 年份 | 核心事件 | 阶段 |
|---|---|---|
| [2012](./2012/) | AlexNet：一声炮响，旧世界终结 | 视觉线 |
| [2013](./2013/) | Word2Vec：词也能有坐标 | 语言线 |
| [2014](./2014/) | GAN、Seq2Seq、Attention、Adam：一年四响 | 生成与序列 |
| [2015](./2015/) | ResNet 与 Batch Norm：深度的解放 | 视觉线 |
| [2016](./2016/) | AlphaGo：强化学习登台 | 决策智能 |
| [2017](./2017/) | Transformer：把 RNN 扔掉 | 语言线 |
| [2018](./2018/) | BERT 与 GPT-1：预训练时代 | 预训练 |
| [2019](./2019/) | GPT-2 与 T5：规模的野心 | 预训练 |
| [2020](./2020/) | GPT-3 与 Scaling Laws：大力出奇迹 | 规模化 |
| [2021](./2021/) | CLIP、Codex、LoRA：多模态与效率 | 多模态 |
| [2022](./2022/) | ChatGPT 与 RLHF：AI 走进大众 | 对齐 |
| [2023](./2023/) | GPT-4 与 LLaMA：开源的反击 | 开源与多模态 |
| [2024](./2024/) | MoE、长上下文、o1：推理时慢思考 | 系统生产 |
| [2025](./2025/) | DeepSeek R1 与 Test-Time Compute：开源追平 | 推理模型 |

## 深度学习前史（2012 之前的基石）

| 年份 | 里程碑 | 继续学 |
|---|---|---|
| [1948](./prehistory/1948-shannon.md) | Shannon 信息论 | [概率与信息论](../foundations/math/probability-information-theory/) |
| [1958](./prehistory/1958-perceptron.md) | Perceptron 感知机 | [深度学习基础](../foundations/deep-learning/deep-learning-basics/) |
| [1986](./prehistory/1986-backprop.md) | Backpropagation 反向传播 | [反向传播](../foundations/deep-learning/backpropagation/) |
| [1997](./prehistory/1997-lstm.md) | LSTM 长短期记忆 | [编码器-解码器](../foundations/structures/encoder-decoder/) |

→ 完整阅读：[prehistory/](./prehistory/)

## 三层架构

| 层 | 入口 | 排序方式 |
|---|---|---|
| L0 · 基础工具箱 | [foundations/](../foundations/) | 按主题树，不按时间 |
| L1 · 编年主线（本目录） | [timeline/](.) | 按年份升序 |
| L2 · 主题深挖 | [tracks/](../tracks/) | 按主题深度递进 |

详见 [docs/restructure.md](../docs/restructure.md)。

---

<!-- 本文件由 scripts/sync_timeline_docs.py 从 web/src/data/timeline.ts 生成 -->
