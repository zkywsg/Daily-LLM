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
