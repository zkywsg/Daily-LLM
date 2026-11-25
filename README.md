<div align="center">

<img src=".github/assets/banner.png" alt="Daily-LLM：从零学大模型，每一步都看得见" width="100%" />

# Daily-LLM：从零学大模型

### 从数学到工程的完整骨架：6 层 · 46 个知识块，讲解 · 互动页 · TikTok 图卡，一份源码三种出口

<p>
  <a href="https://zkywsg.github.io/Daily-LLM/"><strong>在线读 / 玩</strong></a>
  ·
  <a href="#topics"><strong>已完成：注意力</strong></a>
  ·
  <a href="#start"><strong>本地运行</strong></a>
  ·
  <a href="#archive"><strong>旧版归档</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>贡献</strong></a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-0F172A.svg?style=flat-square)](LICENSE)

</div>

---

## 这是什么

一个从零开始学机器学习、深度学习、大模型的资料库，目标是把**整条路线**走完，同时服务自学、TikTok 图文和 GitHub 分享。完整骨架见下图，逐项清单与旧版素材对照见 [learn/ROADMAP.md](learn/ROADMAP.md)。

<img src=".github/assets/roadmap.png" alt="完整骨架（6 层 46 个知识块）" width="100%" />

| 层 | 内容 | 已完成 / 总数 |
|---|---|---|
| 0 数学基础 | 线性代数、概率统计、微积分、信息论、优化直觉 | 0 / 5 |
| 1 机器学习基础 | 回归、树与集成、SVM、聚类降维、评估 | 0 / 5 |
| 2 深度学习基础 | 反传、激活、损失、优化器、归一化、正则化；CNN、RNN、词向量 | 0 / 9 |
| 3 大模型核心 | **注意力系列（已完成）**、Transformer 块、MoE、长上下文；BERT/GPT、预训练、微调；RLHF/DPO、推理模型 | 1 / 10（另有 1 项部分覆盖） |
| 4 模态与应用 | ViT、GAN、扩散、世界模型；多模态、语音；RAG、Agent；图神经网络、推荐 | 0 / 10 |
| 5 工程与系统 | 分布式训练、量化、推理优化、推理服务、评测、两个实战项目 | 0 / 7（另有 1 项部分覆盖） |

**目前只完成了注意力系列**，用它把“讲解 + 互动页 + 图卡”的做法跑通；骨架图里虚线框是还没做的部分。待写项里很多在旧版（`_archive/v1/`）有素材，写新版时取材、重写。

每个知识点按五层写：一句话 → 白话 → 手算 → 数学推导 → 代码与练习。每篇都配：

- **讲解**：本仓库的 Markdown，GitHub 上直接读，公式可渲染，不依赖任何托管。
- **互动页**：浏览器里拖一拖、点一点，数字都在页面里实时算。
- **图卡**：TikTok 竖版 PNG（1080×1920），直接用于发布。

> 例子都是教学用的**玩具模型**，不是训练出来的。真实模型的脚本（`attn-08`）写好了但**没在真实模型上运行过**；文献与配置对照摘要或公开资料核对，来源写在每篇末尾。

<a id="topics"></a>

## 已完成：注意力系列（13 篇）

| 篇 | 内容 | 互动页 |
|---|---|---|
| 基础 A | [注意力基础：q/k/v、打分、加权求和](learn/3-llm/01-attention/) | 有 |
| 基础 B | [多头：并行的多套规则](learn/3-llm/02-multihead/) | 有 |
| 基础 C | [掩码与位置](learn/3-llm/03-mask-position/) | 有 |
| 00 | [动机：为什么发明注意力](learn/3-llm/attn-01-motivation/) | 有 |
| 02 | [手算一遍](learn/3-llm/attn-02-hand-calc/) | 有 |
| 03 | [矩阵与形状](learn/3-llm/attn-03-shapes/) | 有 |
| 04 | [softmax 与 √d：梯度推导](learn/3-llm/attn-04-softmax-scale/) | 有 |
| 05 | [多头的内部：头分解与消融](learn/3-llm/attn-05-multihead-deep/) | 有 |
| 06 | [位置编码：正弦、RoPE、ALiBi](learn/3-llm/attn-06-position/) | 有 |
| 07 | [从零实现（NumPy，7 个测试）](learn/3-llm/attn-07-from-scratch/) | — |
| 08 | [真实模型里的注意力（脚本未在真实模型上运行）](learn/3-llm/attn-08-real-model/) | 查看器 |
| 09 | [高效注意力：KV cache、GQA、FlashAttention](learn/3-llm/attn-09-efficient/) | 有 |
| 10 | [综合自测与小项目（27 题）](learn/3-llm/attn-10-selftest/) | 有 |

> 文献与模型配置已对照摘要或公开资料核对，来源写在各篇末尾。示例模型都是教学用的玩具模型，不是训练出来的。

学习顺序、进度与五层路线图见 [learn/README.md](learn/README.md)。

<a id="start"></a>

## 本地运行

**只想读**：直接看上表链接，或打开 <https://zkywsg.github.io/Daily-LLM/>。

**离线整页预览**（需要 Node 20+）：

```bash
cd learn && npm ci && cd ..
node learn/build-preview.mjs      # 生成 learn/preview.html，双击打开
node learn/build-site.mjs         # 生成 learn/_site/，与线上站点一致
```

**跑代码**：`attn-07` 的从零实现只需 `numpy`（`pip install numpy`），测试在该目录的 `code/` 下；`attn-08` 的两个脚本需要 `torch` 与 `transformers`，未在真实模型上运行过。

互动页是纯前端，也可以直接双击 `learn/3-llm/<知识点>/demo/index.html`。

<a id="archive"></a>

## 旧版归档

早期按“19 个架构家族”写的时间线内容（1986–2025）已于 2026-11-26 整体移入 [`_archive/v1/`](_archive/v1/)，只读，不再更新，新内容以 `learn/` 为准。更早的素材在 [`_archive/`](_archive/README.md) 下。

<details>
<summary>旧版 19 个家族目录</summary>

| # | 家族 | 关键年份 | 一句话定位 |
|---|------|---------|-----------|
| 01 | [CNN 卷积神经网络](_archive/v1/01-cnn/) | 2012– | 把视觉特征从手工设计交给反向传播 |
| 02 | [RNN / LSTM / GRU](_archive/v1/02-rnn-lstm/) | 1997, 2014– | 给神经网络装上"记忆" |
| 03 | [Word Embedding](_archive/v1/03-word-embedding/) | 2013– | 让"词"有了分布式的语义坐标 |
| 04 | [GAN](_archive/v1/04-gan/) | 2014– | 用对抗博弈学会"生成" |
| 05 | [Transformer](_archive/v1/05-transformer/) | 2017 | 用纯注意力替代循环，彻底并行 |
| 06 | [BERT 系预训练](_archive/v1/06-bert-family/) | 2018– | 双向预训练 + 微调成为 NLP 新范式 |
| 07 | [GPT 系 + Scaling](_archive/v1/07-gpt-scaling/) | 2018–2020 | 把规模做到底，涌现 Few-shot |
| 08 | [视觉 Transformer (ViT)](_archive/v1/08-vit/) | 2020– | Transformer 反攻视觉 |
| 09 | [多模态对齐](_archive/v1/09-multimodal-clip/) | 2021– | 让"图"和"文"住进同一个空间 |
| 10 | [扩散模型](_archive/v1/10-diffusion/) | 2020– | 生成的新王，从噪声到图像 |
| 11 | [PEFT / LoRA](_archive/v1/11-peft-lora/) | 2021– | 把大模型微调成本压到普通人能玩 |
| 12 | [对齐与 RLHF](_archive/v1/12-rlhf-alignment/) | 2022– | 把"会答"变成"答得好" |
| 13 | [MoE 与高效推理](_archive/v1/13-moe-efficient/) | 2023– | 大模型在不变贵的前提下变更大 |
| 14 | [RAG 与 Agent](_archive/v1/14-rag-agent/) | 2023– | 把模型接上外部世界 |
| 15 | [推理模型 (o1/R1)](_archive/v1/15-reasoning-o1-r1/) | 2024– | 推理时多想几步，能力再上一个台阶 |
| 16 | [世界模型 / 视频生成](_archive/v1/16-world-models/) | 2018– | 学习环境动态，从想象训练走向可交互视频世界 |
| 17 | [图神经网络](_archive/v1/17-graph-neural-networks/) | 2017– | 把深度学习扩展到任意图结构 |
| 18 | [语音 / 音频模型](_archive/v1/18-speech-audio/) | 2020– | 从自监督语音表征走向音频 token 生成 |
| 19 | [推荐系统](_archive/v1/19-recommendation/) | 2016– | 从特征记忆走向兴趣注意力与图表征学习 |

按年份速查：[TIMELINE.md](_archive/v1/TIMELINE.md) · 横切基础：[foundations/](_archive/v1/foundations/) · 英文版：[README_EN.md](_archive/v1/README_EN.md)

</details>

---

## 贡献与许可

欢迎指出错误、补充内容。提交前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（部分约定针对旧版结构，以 `learn/` 现状为准）。

本项目采用 [MIT License](LICENSE)。
