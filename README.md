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

## 时间线：被逼出来的历史（2012–2025）

| 年份 | 核心突破 | 之前卡在哪 |
|------|---------|-----------|
| 2012 | **AlexNet** — Top-5 错误率 15.3%，比第二名低 11 个百分点 | 视觉特征靠手工设计（SIFT/HOG），识别率多年停滞在 25–26% |
| 2013 | **Word2Vec** — `king − man + woman ≈ queen` | One-Hot 无语义，"猫"与"狗"的距离等于"猫"与"飞机" |
| 2014 | **GAN + Seq2Seq + Attention + Adam** — 四个基础件同年到位 | 模型只能分类不能生成；翻译靠规则对齐表；学习率需手动调 |
| 2015 | **ResNet + Batch Norm** — 152 层，Top-5 低于人类水平 | 网络超 20 层后训练反而变差（退化问题） |
| 2016 | **AlphaGo** — 4:1 击败世界冠军李世石 | 围棋搜索空间天文数字，专家预测 AI 至少还需十年 |
| 2017 | **Transformer** — 纯 Attention 取代 RNN，完全并行 | LSTM 天生串行，句子越长训练越慢 |
| 2018 | **BERT + GPT-1** — 预训练 + 微调范式，一词多义解决 | 静态词向量每词只有一个表示，下游任务需从头训练 |
| 2019 | **GPT-2 + T5** — 1.5B 参数，NLP 任务统一为文本到文本 | BERT 路线内卷，无人知道单纯放大模型会发生什么 |
| 2020 | **GPT-3 + Scaling Laws** — 175B 参数涌现 Few-shot 能力 | 普遍认为大模型必须在每个任务上微调才有效 |
| 2021 | **CLIP + Codex + LoRA** — 图文对齐 / 代码生成 / 低成本微调 | 视觉与语言完全割裂；大模型微调只有巨头能做 |
| 2022 | **ChatGPT + RLHF** — 5 天百万用户，史上最快消费应用增长 | GPT-3 是文本补全工具；无对齐机制，有害内容照单全说 |
| 2023 | **GPT-4 + LLaMA** — 多模态推理质变，开源社区全面爆发 | 大模型权重是少数闭源公司专利，研究者无法触及 |
| 2024 | **MoE + 长上下文 + o1** — 激活参数比例下降；推理时慢思考 | 大模型推理成本线性上涨；复杂推理一步错满盘皆输 |
| 2025 | **DeepSeek R1 + Test-Time Compute** — 开源追平闭源推理能力 | 推理模型是 OpenAI 独门武器；"只有砸钱才能做"无人挑战 |

→ 完整展开（发生了什么 · 解决了什么 · 每年 10 个关键工作）见 [00-Timeline/](00-Timeline/)

---

<a id="modules"></a>

## 模块索引

| Phase | 主题 | 时间段 | 入口 |
|-------|------|--------|------|
| 00 | Timeline — 被逼出来的历史 | 2012–2025 | [00-Timeline/](00-Timeline/) |
| 00 | 前置准备 — 神经网络基础 | — | [00-Prerequisites/](00-Prerequisites/) |
| 01 | 视觉线 — AlexNet → ResNet → GAN | 2012–2017 | [01-Visual-Intelligence/](01-Visual-Intelligence/) |
| 02 | 语言线 — Word2Vec → Transformer → BERT | 2013–2019 | [02-Language-Transformers/](02-Language-Transformers/) |
| 03 | 汇流：规模与多模态 — GPT-3 · ViT · CLIP | 2020–2021 | [03-Scale-Multimodal/](03-Scale-Multimodal/) |
| 04 | 对齐与开源 — RLHF · DPO · LLaMA | 2022–2023 | [04-Alignment-OpenSource/](04-Alignment-OpenSource/) |
| 05 | 系统与生产 — RAG · Agent · vLLM · MLOps | 2023–2025 | [05-Systems-Production/](05-Systems-Production/) |
| 06 | 实战项目 — 企业级端到端系统 | 跨阶段 | [06-Capstone-Projects/](06-Capstone-Projects/) |

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
