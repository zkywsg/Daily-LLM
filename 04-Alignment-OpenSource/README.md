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

**上一阶段**: [汇流：规模与多模态 ←](../03-Scale-Multimodal/) | **下一阶段**: [系统与生产 →](../05-Systems-Production/)
