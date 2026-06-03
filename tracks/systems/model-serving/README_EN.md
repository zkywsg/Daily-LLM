# Model Serving

[English](README_EN.md) | [中文](README.md)

## Where does this problem come from?

> In 2023, vLLM and PagedAttention multiplied LLM inference throughput by an order of magnitude, making model serving an independent and critical engineering domain—it determines the latency, cost, and availability that users actually experience.

## Learning Objectives

After completing this module, you should be able to answer:
1. What are the core advantages of vLLM's PagedAttention over traditional KV-cache management?
2. How do quantization methods like AWQ and GPTQ trade off accuracy vs. speed?
3. Why can Speculative Decoding significantly reduce inference latency?

## Module Contents

- [Serving Architecture](architecture/README.md)
- [Model Compression & Quantization](compression/README.md)
- [Model Registry & Versioning](registry/README.md)

## Evolution Notes

> The legacy of this technology: serving techniques moved large models from "they run" to "they run fast and cheap," but memory explosion under long contexts, batch-scheduling fairness, and multi-model colocation remain active research directions.
→ See [Monitoring & Observability](../monitoring/README.md)

---

**Previous**: [Training Infrastructure](../training-infrastructure/README.md) | **Next**: [Monitoring & Observability](../monitoring/README.md)
