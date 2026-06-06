# Training Infrastructure

[English](README_EN.md) | [中文](README.md)

## Where does this problem come from?

> As model scales broke through tens and hundreds of billions, training was no longer something a single GPU script could handle. Data pipelines, distributed strategies, training stability, and reproducibility became the infrastructure layer for large-model engineering.

## Learning Objectives

After completing this module, you should be able to answer:
1. How do you choose between FSDP and DeepSpeed for different model scales?
2. How do you design a high-throughput, low-failure data pipeline?
3. How do you handle common loss spikes and numerical instability in large-model training?

## Module Contents

- [Distributed Training](distributed/README.md)
- [Data Pipeline](data-pipeline/README.md)
- [PEFT Infrastructure](peft/README.md)
- [Hyperparameter Optimization](hpo/README.md)
- [Training Stability](stability/README.md)
- [Reproducibility](reproducibility/README.md)

## Evolution Notes

> The legacy of this technology: training infrastructure turned "stacking GPUs" into systems engineering, but cluster scheduling, fault recovery, and cross-framework compatibility remain focal points of industrial investment.
→ See [Model Serving](../model-serving/README.md)

---

**Previous**: [Production Systems](../production/README.md) | **Next**: [Model Serving](../model-serving/README.md)
