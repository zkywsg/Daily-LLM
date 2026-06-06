<div align="center">

<img src=".github/assets/readme-banner.svg" alt="Daily-LLM banner" width="100%" />

# 深度学习与大模型精通之路

### Deep Learning & LLM Mastery

<p>
  深度学习与大模型的完整工程路线——从 2012 年 AlexNet 到 2025 年推理模型，每一个技术都是被前一代局限逼出来的。
</p>

<p>
  <a href="README_EN.md"><strong>English</strong></a>
  ·
  <a href="#timeline"><strong>时间线</strong></a>
  ·
  <a href="#web-timeline"><strong>可视化网页</strong></a>
  ·
  <a href="#modules"><strong>模块索引</strong></a>
  ·
  <a href="#quick-start"><strong>快速开始</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>贡献指南</strong></a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-0F172A.svg?style=flat-square)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-2563EB.svg?style=flat-square)](https://www.python.org/downloads/)
[![Phases](https://img.shields.io/badge/Phases-6%2BTimeline-059669.svg?style=flat-square)](./)
[![Bilingual](https://img.shields.io/badge/Docs-English%20%7C%20%E4%B8%AD%E6%96%87-D97706.svg?style=flat-square)](README_EN.md)

</div>

---

<a id="timeline"></a>

## 时间线：15 个家族（2012–2025）

| # | 家族 | 关键年份 | 一句话定位 |
|---|------|---------|-----------|
| 01 | [CNN 卷积神经网络](01-cnn/) | 2012– | 把视觉特征从手工设计交给反向传播 |
| 02 | [RNN / LSTM / GRU](02-rnn-lstm/) | 1997, 2014– | 给神经网络装上"记忆" |
| 03 | [Word Embedding](03-word-embedding/) | 2013– | 让"词"有了分布式的语义坐标 |
| 04 | [GAN](04-gan/) | 2014– | 用对抗博弈学会"生成" |
| 05 | [Transformer](05-transformer/) | 2017 | 用纯注意力替代循环，彻底并行 |
| 06 | [BERT 系预训练](06-bert-family/) | 2018– | 双向预训练 + 微调成为 NLP 新范式 |
| 07 | [GPT 系 + Scaling](07-gpt-scaling/) | 2018–2020 | 把规模做到底，涌现 Few-shot |
| 08 | [视觉 Transformer (ViT)](08-vit/) | 2020– | Transformer 反攻视觉 |
| 09 | [多模态对齐](09-multimodal-clip/) | 2021– | 让"图"和"文"住进同一个空间 |
| 10 | [扩散模型](10-diffusion/) | 2020– | 生成的新王，从噪声到图像 |
| 11 | [PEFT / LoRA](11-peft-lora/) | 2021– | 把大模型微调成本压到普通人能玩 |
| 12 | [对齐与 RLHF](12-rlhf-alignment/) | 2022– | 把"会答"变成"答得好" |
| 13 | [MoE 与高效推理](13-moe-efficient/) | 2023– | 大模型在不变贵的前提下变更大 |
| 14 | [RAG 与 Agent](14-rag-agent/) | 2023– | 把模型接上外部世界 |
| 15 | [推理模型 (o1/R1)](15-reasoning-o1-r1/) | 2024– | 推理时多想几步，能力再上一个台阶 |

> 按年份速查：[TIMELINE.md](TIMELINE.md)（自动生成）
> 横切基础：[foundations/](foundations/)

---

<a id="web-timeline"></a>

## 可视化网页

仓库内已提供一个独立的时间线可视化网页，入口在 [web/](web/)。

本地运行：

```bash
cd web
npm install
npm run dev
```

页面结构是“顶部横向时间线 + 下方当前节点内容”，适合浏览整条深度学习与大模型演进链路。当前先作为本地网页维护，公网部署后再补充正式访问地址。

---

<a id="modules"></a>

## 仓库结构

- `01-cnn/` … `15-reasoning-o1-r1/` — 15 个架构/范式家族，按登场时间排序
- `foundations/` — 横切基础（激活、反传、优化器、归一化、注意力机制…）
- `projects/` — 跨家族实战项目
- `web/` — 可视化网页
- `TIMELINE.md` — 自动生成的按年份索引
- `_archive/` — 旧 `timeline/` 与 `tracks/` 内容，作为家族内容重写时的素材源

---

<a id="quick-start"></a>

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/zkywsg/Daily-LLM.git
cd Daily-LLM
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>按阶段选择性安装依赖</strong></summary>

```bash
# Phase 00-01（前置 + 视觉线）
pip install torch numpy scikit-learn matplotlib

# Phase 02-03（语言线 + 规模多模态）
pip install transformers datasets sentence-transformers

# Phase 04（对齐与微调）
pip install peft trl

# Phase 05（系统与生产）
pip install sentence-transformers faiss-cpu chromadb langchain vllm fastapi mlflow wandb
```

</details>

---

## 贡献

欢迎贡献改进内容、补充案例、修正文档或完善结构。提交前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。
