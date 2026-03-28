# Phase 04: LLM 核心

> 从"预测下一个词"到"理解人类意图"，再到"推理时多想一会儿"。
> 这一阶段是整个课程的核心：大语言模型如何被训练、对齐、高效使用。

## 为什么这一阶段是核心

Phase 03 建立了 Transformer 和预训练的基础。这一阶段进入 LLM 的核心工程问题：

- GPT-3 证明了 **Scale = 能力**，但训练成本只有极少数机构能承担
- LoRA / QLoRA 让**个人也能微调大模型**，开源生态由此爆发
- RLHF / DPO 解决了**对齐问题**：从"预测下一个词"变成"学习让人满意"
- o1 开辟了**推理时 Scaling** 的新方向：用更多计算换更高准确率

## 本阶段内容

### 1. [预训练](pre-training/README.md)
大模型的训练流程与 Scaling Laws。
- 数据工程：清洗、去重、配比
- 架构选择：GPT vs BERT vs T5，为什么 Decoder-Only 胜出
- Scaling Laws：参数、数据、算力的幂律关系
- Chinchilla 最优：训练充分性的量化标准

### 2. [高效微调 PEFT](peft/README.md)
用 1/1000 的参数量达到全量微调的效果。
- LoRA：低秩矩阵旁路，原理与实现
- QLoRA：量化 + LoRA，在消费级 GPU 上微调 70B
- Adapter、Prefix Tuning、Prompt Tuning 对比
- 实战：使用 HuggingFace PEFT 库

### 3. [对齐](alignment/README.md)
让模型"有用、无害、诚实"。
- SFT：监督微调，用人类示范数据
- RLHF：奖励模型 + PPO，三步流程详解
- DPO：直接偏好优化，更简洁的对齐方案
- Constitutional AI：用 AI 反馈替代人工标注

### 4. [提示工程](prompt-engineering/README.md)
最大化利用模型的 In-Context Learning 能力。
- Zero-shot / Few-shot / Chain-of-Thought
- ReAct：推理 + 行动的结合
- 结构化输出与 JSON Mode
- 提示注入风险与防护

### 5. [框架](frameworks/README.md)
主流 LLM 训练与推理框架全景。
- 训练：Megatron-LM、DeepSpeed、FSDP
- 推理：vLLM、TGI、TensorRT-LLM
- 应用：LangChain、LlamaIndex、OpenAI SDK

### 6. [多模态](multimodal/README.md)
视觉 + 语言的统一理解与生成。
- CLIP：对比学习的多模态表示
- 视觉语言模型：LLaVA、Flamingo、GPT-4V
- 扩散模型基础：DALL-E、Stable Diffusion
- 视频生成：Sora 的技术路线

## 时间线节点

本模块对应时间线中的以下关键工作：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2020 | GPT-3 | 175B 参数，In-Context Learning 涌现，Scale 即能力 |
| 2020 | Scaling Laws | 参数/数据/算力的幂律关系，大模型训练从玄学变工程 |
| 2020 | ViT | 纯 Transformer 处理图像，多模态统一架构的基础 |
| 2020 | Switch Transformer | 万亿参数 MoE，条件计算降低推理成本 |
| 2021 | CLIP | 图文对比学习，多模态理解的关键里程碑 |
| 2021 | Codex | GitHub Copilot 背后，代码生成实用化 |
| 2021 | LoRA | 低秩微调，让个人可以微调大模型 |
| 2021 | FLAN | 指令微调，Zero-shot 能力大幅提升 |
| 2022 | ChatGPT / RLHF | 三步对齐流程，模型从"补全文本"到"理解意图" |
| 2022 | Chinchilla | 训练充分性定律，数据 ≥ 参数同等重要 |
| 2022 | DPO | 直接偏好优化，比 RLHF 更简洁稳定 |
| 2022 | Flash Attention | 显存 IO 优化，训练速度 2-4×，长上下文成为可能 |
| 2023 | GPT-4 | 多模态推理，质的飞跃 |
| 2023 | LLaMA / LLaMA 2 | 开源大模型，社区生态全面爆发 |
| 2023 | Mistral 7B | 效率极致，7B 打赢 13B |
| 2024 | o1 | 推理时 Scaling，慢思考换高准确率 |
| 2024 | Mixtral 8x7B | MoE 主流化开源 |
| 2025 | DeepSeek R1 | 纯 RL 训练推理能力，开源追平闭源 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)
