# Phase 03 · Convergence: Scale & Multimodal (2020–2021)

[English](README_EN.md) | [中文](README.md)

The two lines converge here. Transformer conquered both language and vision,
Scale was proven to be an emergent capability, and image-text understanding was unified in a single space for the first time.

## Modules

### [Pre-training & Scale](pre-training/README_EN.md)
GPT-3, Scaling Laws, data engineering
- Scaling Laws: power-law relationships among parameters, data, and compute
- Chinchilla optimality: quantitative standards for training sufficiency
- Data engineering: cleaning, deduplication, mixture ratios
- ViT: Transformer for images (the vision line concludes here)

### [Multimodal](multimodal/README_EN.md)
CLIP → DALL-E → Codex: vision and language formally unified
- CLIP: contrastive learning for image-text alignment
- DALL-E: text-to-image generation
- Flamingo / Perceiver IO: multimodal few-shot learning
- Codex: code generation, powering GitHub Copilot

### [Prompt Engineering](prompt-engineering/README_EN.md)
Systematic methodology for In-Context Learning
- Zero-shot / Few-shot / Chain-of-Thought
- ReAct reasoning + action framework
- Structured output

### [Frameworks](frameworks/README_EN.md)
Landscape of mainstream training frameworks
- Megatron-LM, DeepSpeed, FSDP
- HuggingFace Transformers ecosystem

## Timeline Nodes

| Year | Work | Core Significance |
|------|------|-------------------|
| 2020 | GPT-3 + Scaling Laws | 175B parameters enable Few-shot; scale becomes predictable |
| 2020 | ViT | Transformer handles images; vision line concludes |
| 2020 | RAG paper | First combination of retrieval + generation, addressing hallucination and knowledge cutoff |
| 2021 | CLIP | Unified image-text representation; the two lines formally converge |
| 2021 | DALL-E | Text-to-image generation enters public awareness |
| 2021 | Codex | Code generation becomes practical; GitHub Copilot ships |
| 2021 | LoRA | Low-rank fine-tuning makes personal LLM tuning possible |
| 2021 | FLAN | Instruction fine-tuning dramatically improves zero-shot capability |

→ Full timeline: [../../timeline](../../timeline/)

**Previous Phase**: [Language & Transformers ←](../language/) | **Next Phase**: [Alignment & Open Source →](../alignment/)
