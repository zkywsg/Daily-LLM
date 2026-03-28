# Project Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 Phase 01–07 重组为新的 7 Phase 结构（视觉线 + 语言线 → 汇流），并重写根 README 去除所有营销性内容，以时间线表格为核心。

**Architecture:** 使用 `git mv` 保留文件历史，按任务逐步重命名和迁移目录，每步提交一次。README 文件在目录结构确定后统一重写。

**Tech Stack:** git, markdown

---

## 目录映射总览

```
旧结构                          新结构
─────────────────────────────────────────────────────
00-Timeline/              →  00-Timeline/（不变）
01-Foundations/           →  00-Prerequisites/
  machine-learning/       →  删除
  deep-learning-basics/   →  保留（重命名 README）
02-Neural-Networks/       →  01-Visual-Intelligence/
  cnn-architectures/      →  保留
  sequence-models/        →  保留
  training/               →  保留
03-NLP-Transformers/      →  02-Language-Transformers/
  attention-mechanisms/   →  保留
  pretrained-models/      →  保留
  transformer-architecture/ → 保留
04-LLM-Core/              →  03-Scale-Multimodal/
  pre-training/           →  保留
  multimodal/             →  保留
  prompt-engineering/     →  保留
  frameworks/             →  保留
  alignment/              →  移至 04-Alignment-OpenSource/
  peft/                   →  移至 04-Alignment-OpenSource/
新建：04-Alignment-OpenSource/
  alignment/              ←  从 04-LLM-Core 迁入
  peft/                   ←  从 04-LLM-Core 迁入
05-RAG-Systems/           →  05-Systems-Production/
  rag-foundations/        →  保留
  vector-databases/       →  保留
  agents/                 →  保留
  production/             →  保留
06-MLOps-Production/      →  合并进 05-Systems-Production/
  training-infrastructure/ → 05-Systems-Production/training-infrastructure/
  model-serving/          → 05-Systems-Production/model-serving/
  monitoring/             → 05-Systems-Production/monitoring/
  deployment/             → 05-Systems-Production/deployment/
07-Capstone-Projects/     →  06-Capstone-Projects/
  enterprise-rag-system/  →  保留
  finetune-deploy-pipeline/ → 保留
```

---

## Task 1: 重命名 01-Foundations → 00-Prerequisites，删除 machine-learning

**Files:**
- Rename: `01-Foundations/` → `00-Prerequisites/`
- Delete: `00-Prerequisites/machine-learning/`

- [ ] **Step 1: git mv 重命名目录**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git mv 01-Foundations 00-Prerequisites
```

Expected: no error, git tracks the rename

- [ ] **Step 2: 删除 machine-learning 子目录**

```bash
git rm -r 00-Prerequisites/machine-learning
```

Expected: removes `machine-learning/README.md` and `machine-learning/README_EN.md`

- [ ] **Step 3: 验证结构**

```bash
ls 00-Prerequisites/
```

Expected output:
```
README.md  README_EN.md  deep-learning-basics/
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename Foundations→Prerequisites, remove classical ML content"
```

---

## Task 2: 重命名 02-Neural-Networks → 01-Visual-Intelligence

**Files:**
- Rename: `02-Neural-Networks/` → `01-Visual-Intelligence/`

- [ ] **Step 1: git mv**

```bash
git mv 02-Neural-Networks 01-Visual-Intelligence
```

- [ ] **Step 2: 验证**

```bash
ls 01-Visual-Intelligence/
```

Expected output:
```
README.md  cnn-architectures/  sequence-models/  training/
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: rename Neural-Networks → Visual-Intelligence"
```

---

## Task 3: 重命名 03-NLP-Transformers → 02-Language-Transformers

**Files:**
- Rename: `03-NLP-Transformers/` → `02-Language-Transformers/`

- [ ] **Step 1: git mv**

```bash
git mv 03-NLP-Transformers 02-Language-Transformers
```

- [ ] **Step 2: 验证**

```bash
ls 02-Language-Transformers/
```

Expected output:
```
README.md  attention-mechanisms/  pretrained-models/  transformer-architecture/
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: rename NLP-Transformers → Language-Transformers"
```

---

## Task 4: 重命名 04-LLM-Core → 03-Scale-Multimodal，移出 alignment 和 peft

**Files:**
- Rename: `04-LLM-Core/` → `03-Scale-Multimodal/`
- Move out: `03-Scale-Multimodal/alignment/` → 暂存（Task 5 移入新目录）
- Move out: `03-Scale-Multimodal/peft/` → 暂存（Task 5 移入新目录）

- [ ] **Step 1: git mv 重命名**

```bash
git mv 04-LLM-Core 03-Scale-Multimodal
```

- [ ] **Step 2: 验证当前内容**

```bash
ls 03-Scale-Multimodal/
```

Expected output:
```
README.md  alignment/  frameworks/  multimodal/  peft/  pre-training/  prompt-engineering/
```

- [ ] **Step 3: Commit 重命名（在移动子目录前先提交，保留历史）**

```bash
git add -A
git commit -m "refactor: rename LLM-Core → Scale-Multimodal"
```

---

## Task 5: 创建 04-Alignment-OpenSource，迁入 alignment 和 peft

**Files:**
- Create dir: `04-Alignment-OpenSource/`
- Move in: `03-Scale-Multimodal/alignment/` → `04-Alignment-OpenSource/alignment/`
- Move in: `03-Scale-Multimodal/peft/` → `04-Alignment-OpenSource/peft/`

- [ ] **Step 1: git mv alignment**

```bash
git mv 03-Scale-Multimodal/alignment 04-Alignment-OpenSource/alignment
```

- [ ] **Step 2: git mv peft**

```bash
git mv 03-Scale-Multimodal/peft 04-Alignment-OpenSource/peft
```

- [ ] **Step 3: 验证两个目录**

```bash
ls 03-Scale-Multimodal/
```

Expected output:
```
README.md  frameworks/  multimodal/  pre-training/  prompt-engineering/
```

```bash
ls 04-Alignment-OpenSource/
```

Expected output:
```
alignment/  peft/
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: create Alignment-OpenSource, move alignment+peft from Scale-Multimodal"
```

---

## Task 6: 重命名 05-RAG-Systems → 05-Systems-Production，合并 06-MLOps-Production

**Files:**
- Rename: `05-RAG-Systems/` → `05-Systems-Production/`
- Move in: `06-MLOps-Production/training-infrastructure/` → `05-Systems-Production/training-infrastructure/`
- Move in: `06-MLOps-Production/model-serving/` → `05-Systems-Production/model-serving/`
- Move in: `06-MLOps-Production/monitoring/` → `05-Systems-Production/monitoring/`
- Move in: `06-MLOps-Production/deployment/` → `05-Systems-Production/deployment/`
- Delete: `06-MLOps-Production/` (empty after moves)

- [ ] **Step 1: git mv 重命名**

```bash
git mv 05-RAG-Systems 05-Systems-Production
```

- [ ] **Step 2: Commit 重命名**

```bash
git add -A
git commit -m "refactor: rename RAG-Systems → Systems-Production"
```

- [ ] **Step 3: 迁移 06-MLOps-Production 的四个子目录**

```bash
git mv 06-MLOps-Production/training-infrastructure 05-Systems-Production/training-infrastructure
git mv 06-MLOps-Production/model-serving 05-Systems-Production/model-serving
git mv 06-MLOps-Production/monitoring 05-Systems-Production/monitoring
git mv 06-MLOps-Production/deployment 05-Systems-Production/deployment
```

- [ ] **Step 4: 删除 06-MLOps-Production（只剩 README 等文件）**

```bash
git rm -r 06-MLOps-Production
```

- [ ] **Step 5: 验证**

```bash
ls 05-Systems-Production/
```

Expected output:
```
README.md  README_EN.md  agents/  deployment/  model-serving/
monitoring/  production/  rag-foundations/  training-infrastructure/  vector-databases/
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: merge MLOps-Production into Systems-Production"
```

---

## Task 7: 重命名 07-Capstone-Projects → 06-Capstone-Projects

**Files:**
- Rename: `07-Capstone-Projects/` → `06-Capstone-Projects/`

- [ ] **Step 1: git mv**

```bash
git mv 07-Capstone-Projects 06-Capstone-Projects
```

- [ ] **Step 2: 验证整体结构**

```bash
ls /Users/lauzanhing/Desktop/Daily-LLM/
```

Expected output:
```
00-Prerequisites/  00-Timeline/  01-Visual-Intelligence/  02-Language-Transformers/
03-Scale-Multimodal/  04-Alignment-OpenSource/  05-Systems-Production/
06-Capstone-Projects/  CHANGELOG.md  CLAUDE.md  CONTRIBUTING.md
LICENSE  README.md  README_EN.md  STYLE.md  docs/  requirements.txt
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: rename Capstone-Projects to 06-Capstone-Projects"
```

---

## Task 8: 重写各 Phase 根 README

每个 Phase 一个 README，格式统一：标题 + 一句话定位 + 时间线节点表 + 子模块列表。

**Files:**
- Modify: `00-Prerequisites/README.md`
- Modify: `01-Visual-Intelligence/README.md`
- Modify: `02-Language-Transformers/README.md`
- Modify: `03-Scale-Multimodal/README.md`
- Create: `04-Alignment-OpenSource/README.md`
- Modify: `05-Systems-Production/README.md`
- Modify: `06-Capstone-Projects/README.md`

- [ ] **Step 1: 重写 00-Prerequisites/README.md**

内容如下（完整替换）：

```markdown
# Phase 00 · 前置准备

进入视觉线和语言线之前所需的最低限度神经网络基础。
不包含经典机器学习算法（SVM、决策树、K-Means 等）。

## 本阶段内容

### [深度学习基础](deep-learning-basics/README.md)
- 神经元与前向传播
- 反向传播与梯度下降
- 激活函数：Sigmoid、ReLU 及其变体
- 损失函数与训练循环
- 过拟合与正则化基础

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2012 | ReLU 普及 | 取代 sigmoid，梯度流动恢复，训练速度提升数倍 |
| 2012 | GPU 深度学习生态 | CUDA 加速训练，计算基础设施确立 |
| 2014 | Adam 优化器 | 几乎不需要调学习率的默认优化器 |
| 2015 | Batch Normalization | 训练速度提升数量级，允许更高学习率 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [视觉线 →](../01-Visual-Intelligence/)
```

- [ ] **Step 2: 重写 01-Visual-Intelligence/README.md**

```markdown
# Phase 01 · 视觉线（2012–2017）

从手工特征到自动学习，CNN 如何一步步解放视觉理解的上限。
这条线在 2020 年 ViT 处与语言线汇流。

## 本阶段内容

### [CNN 架构](cnn-architectures/README.md)
AlexNet → ZFNet → VGGNet → GoogLeNet → ResNet → DenseNet
- 卷积、池化、感受野
- 深度 vs 宽度的系统探索
- 跳跃连接解决退化问题
- 轻量化架构：MobileNet、SqueezeNet、SE-Net

### [序列模型](sequence-models/README.md)
GAN、VAE、WaveNet，以及 AlphaGo（CNN + RL）
- 生成对抗网络的对抗训练原理
- 变分自编码器的潜变量空间
- 自回归波形生成
- 强化学习与 CNN 的结合

### [训练与优化](training/README.md)
Dropout、Batch Norm、数据增强、GPU 训练技巧
- 正则化：Dropout、DropConnect
- 归一化：Batch Norm、Layer Norm
- 优化器：SGD、Adam
- 训练稳定性工程

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2012 | AlexNet | 手工特征时代终结，深度学习元年 |
| 2013 | ZFNet / VAE | CNN 可视化；连续潜变量生成模型 |
| 2014 | VGGNet / GoogLeNet / GAN | 深度探索；多尺度架构；生成对抗训练 |
| 2015 | ResNet / Batch Norm | 152 层，Top-5 低于人类；训练速度飞跃 |
| 2016 | DenseNet / AlphaGo / WaveNet | 特征复用；CNN+RL 决策；自回归音频生成 |
| 2017 | SE-Net / Progressive GAN | 通道注意力；渐进式高质量图像生成 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [语言线 →](../02-Language-Transformers/)
```

- [ ] **Step 3: 重写 02-Language-Transformers/README.md**

```markdown
# Phase 02 · 语言线（2013–2019）

从词是坐标，到上下文决定含义，再到注意力就是一切。
这条线在 2021 年 CLIP 处与视觉线汇流。

## 本阶段内容

### [注意力机制](attention-mechanisms/README.md)
从 Bahdanau Attention 到 Self-Attention
- Seq2Seq 的信息瓶颈问题
- Bahdanau / Luong Attention 数学推导
- Self-Attention 与 Multi-Head Attention
- 位置编码

### [Transformer 架构](transformer-architecture/README.md)
《Attention Is All You Need》完整拆解
- Encoder / Decoder 结构
- 残差连接 + Layer Normalization
- 训练技巧：Warm-up、Label Smoothing

### [预训练模型](pretrained-models/README.md)
BERT、GPT、T5 三条路线系统对比
- ELMo：上下文动态词向量
- GPT-1/2：单向自回归预训练
- BERT：双向 Masked LM
- T5：统一文本到文本框架
- 变体：RoBERTa、ALBERT、DistilBERT、XLNet

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2013 | Word2Vec | 词向量起点，语义空间坐标系 |
| 2014 | GloVe / Seq2Seq / Attention | 全局共现词向量；端到端翻译；信息瓶颈的第一个解法 |
| 2016 | FastText | 子词级词向量，OOV 问题解决 |
| 2017 | Transformer | 纯 Attention 取代 RNN，完全并行 |
| 2018 | ELMo / GPT-1 / BERT | 动态词向量；预训练 + 微调范式确立 |
| 2019 | GPT-2 / T5 / RoBERTa | 规模化生成；任务统一；BERT 训练不足被证明 |
| 2019 | ALBERT / DistilBERT | 参数共享压缩；知识蒸馏轻量化 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [汇流：规模与多模态 →](../03-Scale-Multimodal/)
```

- [ ] **Step 4: 重写 03-Scale-Multimodal/README.md**

```markdown
# Phase 03 · 汇流：规模与多模态（2020–2021）

两条线在这里合并。Transformer 同时征服了语言和视觉，
Scale 被证明是一种涌现能力，图文理解第一次统一在同一个空间里。

## 本阶段内容

### [预训练与规模](pre-training/README.md)
GPT-3、Scaling Laws、数据工程
- Scaling Laws：参数 / 数据 / 算力的幂律关系
- Chinchilla 最优：训练充分性的量化标准
- 数据工程：清洗、去重、配比
- ViT：Transformer 处理图像（视觉线在此收尾）

### [多模态](multimodal/README.md)
CLIP → DALL-E → Codex，视觉与语言正式统一
- CLIP：对比学习图文对齐
- DALL-E：文本生成图像
- Flamingo / Perceiver IO：多模态少样本学习
- Codex：代码生成，GitHub Copilot 背后

### [提示工程](prompt-engineering/README.md)
In-Context Learning 的系统方法论
- Zero-shot / Few-shot / Chain-of-Thought
- ReAct 推理 + 行动框架
- 结构化输出

### [框架](frameworks/README.md)
主流训练框架全景
- Megatron-LM、DeepSpeed、FSDP
- HuggingFace Transformers 生态

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2020 | GPT-3 + Scaling Laws | 175B 参数涌现 Few-shot；规模可预测 |
| 2020 | ViT | Transformer 处理图像，视觉线收尾 |
| 2020 | RAG 论文 | 检索 + 生成首次结合，解决幻觉和知识截止 |
| 2021 | CLIP | 图文统一表示，两条线正式汇流 |
| 2021 | DALL-E | 文本生成图像进入公众视野 |
| 2021 | Codex | 代码生成实用化，GitHub Copilot 落地 |
| 2021 | LoRA | 低秩微调，个人可微调大模型 |
| 2021 | FLAN | 指令微调，Zero-shot 能力大幅提升 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [对齐与开源 →](../04-Alignment-OpenSource/)
```

- [ ] **Step 5: 创建 04-Alignment-OpenSource/README.md**

```markdown
# Phase 04 · 对齐与开源（2022–2023）

大模型能用之后，两个问题同时爆发：
怎么让它「听话」，以及怎么让更多人用得起。

## 本阶段内容

### [对齐](alignment/README.md)
从「预测下一个词」到「学习让人满意」
- SFT：监督微调，用人类示范数据
- RLHF：奖励模型 + PPO，三步流程详解
- DPO：直接偏好优化，比 RLHF 更简洁稳定
- Constitutional AI：用 AI 反馈替代人工标注
- Alignment Tax：过度对齐与奖励 Hacking

### [高效微调 PEFT](peft/README.md)
用 1/1000 的参数量达到全量微调的效果
- LoRA：低秩矩阵旁路，原理与实现
- QLoRA：量化 + LoRA，消费级 GPU 微调 70B
- Adapter、Prefix Tuning、Prompt Tuning 对比
- 实战：HuggingFace PEFT 库

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2022 | ChatGPT / InstructGPT | RLHF 三步对齐流程，5 天百万用户 |
| 2022 | Chinchilla | 训练充分性定律，数据 ≥ 参数同等重要 |
| 2022 | DPO | 直接偏好优化，不需要独立奖励模型 |
| 2022 | Constitutional AI | 用 AI 反馈训练 AI，减少人工标注 |
| 2022 | Flash Attention | IO 感知注意力，训练速度 2-4× |
| 2023 | LLaMA / LLaMA 2 | 开源大模型，社区生态全面爆发 |
| 2023 | Mistral 7B | 7B 打赢 13B，效率极致 |
| 2023 | Mixtral 8x7B | MoE 开源，激活参数比例大幅下降 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [系统与生产 →](../05-Systems-Production/)
```

- [ ] **Step 6: 重写 05-Systems-Production/README.md**

```markdown
# Phase 05 · 系统与生产（2023–2025）

从模型能力到真实系统落地。
RAG、Agent、推理优化、生产监控，这一阶段把前四个 Phase 的知识变成可交付的系统。

## 本阶段内容

### [RAG 基础](rag-foundations/README.md)
检索增强生成系统的核心组件
- Chunking 策略、Embedding 模型选型
- Rerank、Query Rewriting、HyDE
- 检索评估：MRR、NDCG

### [向量数据库](vector-databases/README.md)
语义搜索的存储引擎
- HNSW、IVF、标量量化
- Milvus、Pinecone、Weaviate、Chroma

### [智能体](agents/README.md)
构建自主系统
- ReAct、Plan-and-Solve 框架
- 工具调用、记忆、多智能体协作
- LangChain / LlamaIndex 工程化

### [生产系统](production/README.md)
真实世界应用模式
- 企业搜索、代码助手、对话系统

### [训练基础设施](training-infrastructure/README.md)
分布式训练流水线
- FSDP、DeepSpeed、3D 并行
- 数据流水线、训练稳定性

### [模型服务](model-serving/README.md)
高效推理与部署
- vLLM / PagedAttention、TGI、TensorRT-LLM
- 量化：AWQ、GPTQ
- Speculative Decoding

### [监控与可观测性](monitoring/README.md)
确保系统健康与质量
- RAG 可观测性、漂移检测
- 基准测试、评估体系

### [部署基础设施](deployment/README.md)
面向 AI 的 DevOps
- CI/CD for ML、成本优化、Kubernetes

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2023 | RAG 生产化 / LangChain | 检索增强系统工程化框架成型 |
| 2023 | AutoGPT / ReAct | Agent 破圈，多步任务执行 |
| 2023 | vLLM / PagedAttention | LLM 推理吞吐量提升 24× |
| 2024 | o1 / 推理模型 | 推理时慢思考，复杂任务准确率飞跃 |
| 2024 | MoE 主流化 | 激活参数比例下降，推理成本显著降低 |
| 2024 | Flash Attention 2/3 | 支持更长上下文训练 |
| 2025 | DeepSeek R1 | 纯 RL 训练推理能力，开源追平闭源 |
| 2025 | Speculative Decoding 普及 | 草稿模型加速推理 2-3× |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [实战项目 →](../06-Capstone-Projects/)
```

- [ ] **Step 7: 重写 06-Capstone-Projects/README.md**

```markdown
# Phase 06 · 实战项目

把前五个 Phase 的知识组装成可以真实交付的系统。

## 本阶段项目

### [企业级 RAG 系统](enterprise-rag-system/README.md)
支持百万文档检索的生产级 RAG 系统
- 文档处理 → Embedding → 向量存储 → 检索增强 → 生成 → 评估 → 上线监控
- 对应技术：Phase 03（LoRA）+ Phase 04（对齐）+ Phase 05（RAG + vLLM）

### [自动化微调与部署流水线](finetune-deploy-pipeline/README.md)
从数据准备到上线监控的端到端 LLM 微调工程
- 指令数据构造 → LoRA/QLoRA 微调 → DPO 对齐 → 评估 → vLLM 部署 → 监控
- 对应技术：Phase 04（PEFT + 对齐）+ Phase 05（服务 + MLOps）

→ 完整时间线见 [00-Timeline](../00-Timeline/)
```

- [ ] **Step 8: Commit 所有 README**

```bash
git add -A
git commit -m "docs: rewrite all Phase READMEs for new structure"
```

---

## Task 9: 重写根 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 完整替换 README.md 内容**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with timeline table as centerpiece, remove all marketing content"
```

---

## Task 10: 更新 00-Timeline/README.md 中的模块链接

**Files:**
- Modify: `00-Timeline/README.md`

旧的模块链接格式如 `[02·神经网络](../02-Neural-Networks/)` 需要更新为新路径。

- [ ] **Step 1: 批量替换模块链接**

将所有 `同年关键工作速览` 表格中的「所属模块」列链接更新：

| 旧链接 | 新链接 |
|--------|--------|
| `[01·基础](../01-Foundations/)` | `[00·前置](../00-Prerequisites/)` |
| `[02·神经网络](../02-Neural-Networks/)` | `[01·视觉线](../01-Visual-Intelligence/)` |
| `[03·NLP](../03-NLP-Transformers/)` | `[02·语言线](../02-Language-Transformers/)` |
| `[04·LLM核心](../04-LLM-Core/)` | `[03·规模多模态](../03-Scale-Multimodal/)` 或 `[04·对齐开源](../04-Alignment-OpenSource/)` |
| `[05·RAG&Agent](../05-RAG-Systems/)` | `[05·系统生产](../05-Systems-Production/)` |
| `[06·MLOps](../06-MLOps-Production/)` | `[05·系统生产](../05-Systems-Production/)` |

使用编辑器或以下命令批量替换（逐个确认）：

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
# 预览替换结果
grep -n "01-Foundations\|02-Neural-Networks\|03-NLP-Transformers\|04-LLM-Core\|05-RAG-Systems\|06-MLOps-Production\|07-Capstone" 00-Timeline/README.md | head -20
```

然后用编辑器逐条替换（00-Timeline/README.md 共约 15 处链接需更新）。

- [ ] **Step 2: 更新文件末尾的模块对照表**

将末尾的「各模块覆盖时间段」表格更新为新路径：

```markdown
| 模块 | 覆盖的时间线节点 |
|------|----------------|
| [00·前置准备](../00-Prerequisites/) | 神经网络基础 |
| [01·视觉线](../01-Visual-Intelligence/) | 2012–2017 CNN · RNN · 生成模型 |
| [02·语言线](../02-Language-Transformers/) | 2013–2019 词向量 · Attention · BERT |
| [03·规模与多模态](../03-Scale-Multimodal/) | 2020–2021 GPT-3 · ViT · CLIP |
| [04·对齐与开源](../04-Alignment-OpenSource/) | 2022–2023 RLHF · DPO · LLaMA |
| [05·系统与生产](../05-Systems-Production/) | 2023–2025 RAG · Agent · 推理 · MLOps |
| [06·实战项目](../06-Capstone-Projects/) | 跨阶段综合 |
```

- [ ] **Step 3: Commit**

```bash
git add 00-Timeline/README.md
git commit -m "docs: update Timeline module links to new Phase structure"
```

---

## Task 11: 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新项目结构说明**

```markdown
# CLAUDE.md — Daily-LLM 项目指令

## 对话开始时必须做的事

每次对话开始，必须先调用 `using-superpowers` skill，再做任何其他事情（包括回答问题、澄清需求）。

## 项目背景

- **仓库**：Daily-LLM，深度学习与大模型的双语学习知识库
- **主要语言**：中文优先，英文为辅
- **结构**：
  - `00-Timeline/` — 编年体时间线（2012–2025）
  - `00-Prerequisites/` — 神经网络基础前置
  - `01-Visual-Intelligence/` — 视觉线（2012–2017）
  - `02-Language-Transformers/` — 语言线（2013–2019）
  - `03-Scale-Multimodal/` — 汇流：规模与多模态（2020–2021）
  - `04-Alignment-OpenSource/` — 对齐与开源（2022–2023）
  - `05-Systems-Production/` — 系统与生产（2023–2025）
  - `06-Capstone-Projects/` — 实战项目

## 核心原则

### 时间线与模块双向同步
当在 `00-Timeline/README.md` 增加或修改某个条目时，必须同步更新对应模块的 README。
反之亦然：模块新增内容时，检查时间线是否需要补充对应节点。
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with new Phase structure"
```

---

## Task 12: 最终验证

- [ ] **Step 1: 验证目录结构完整**

```bash
ls /Users/lauzanhing/Desktop/Daily-LLM/
```

Expected:
```
00-Prerequisites/  00-Timeline/  01-Visual-Intelligence/  02-Language-Transformers/
03-Scale-Multimodal/  04-Alignment-OpenSource/  05-Systems-Production/
06-Capstone-Projects/  CHANGELOG.md  CLAUDE.md  CONTRIBUTING.md
LICENSE  README.md  README_EN.md  STYLE.md  docs/  requirements.txt
```

- [ ] **Step 2: 验证所有内部链接无断链（关键目录）**

```bash
# 检查 README 中是否有旧路径残留
grep -r "01-Foundations\|02-Neural-Networks\|03-NLP-Transformers\|04-LLM-Core\|05-RAG-Systems\|06-MLOps-Production\|07-Capstone" \
  README.md CLAUDE.md 00-Timeline/README.md
```

Expected: no output（无旧路径残留）

- [ ] **Step 3: 验证 git log**

```bash
git log --oneline -12
```

Expected: 12 commits corresponding to each task above

- [ ] **Step 4: 最终 commit（如有遗漏）**

```bash
git add -A
git status  # 确认 clean working tree
```
