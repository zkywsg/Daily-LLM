# Phase 04 · Alignment & Open Source (2022–2023)

[English](README_EN.md) | [中文](README.md)

After large models became usable, two problems erupted simultaneously:
how to make them "obedient," and how to let more people afford them.

## Modules

### [Alignment](alignment/README_EN.md)
From "predict the next token" to "learn to satisfy humans"
- SFT: supervised fine-tuning with human demonstration data
- RLHF: reward model + PPO, three-step process explained
- DPO: direct preference optimization, simpler and more stable than RLHF
- Constitutional AI: replacing human labels with AI feedback
- Alignment Tax: over-alignment and reward hacking

### [Efficient Fine-tuning PEFT](peft/README_EN.md)
Achieving full fine-tuning effects with 1/1000 of the parameters
- LoRA: low-rank matrix bypass, principles and implementation
- QLoRA: quantization + LoRA, fine-tuning 70B on consumer GPUs
- Comparison of Adapter, Prefix Tuning, and Prompt Tuning
- Hands-on: HuggingFace PEFT library

## Timeline Nodes

| Year | Work | Core Significance |
|------|------|-------------------|
| 2022 | ChatGPT / InstructGPT | RLHF three-step alignment, 1M users in 5 days |
| 2022 | Chinchilla | Training sufficiency law: data ≥ parameters in importance |
| 2022 | DPO | Direct preference optimization without a separate reward model |
| 2022 | Constitutional AI | Training AI with AI feedback, reducing human annotation |
| 2022 | Flash Attention | IO-aware attention, 2–4× training speedup |
| 2023 | LLaMA / LLaMA 2 | Open-source large models; community ecosystem explodes |
| 2023 | Mistral 7B | 7B beats 13B, efficiency pushed to the extreme |
| 2023 | Mixtral 8x7B | Open-source MoE, activated parameter ratio drops sharply |

→ Full timeline: [00-Timeline](../00-Timeline/)

**Previous Phase**: [Convergence: Scale & Multimodal ←](../03-Scale-Multimodal/) | **Next Phase**: [Systems & Production →](../05-Systems-Production/)
