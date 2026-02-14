# Phase 6: MLOps & Production Engineering

**[English](README_EN.md) | [中文](README.md)**

## Overview

This phase covers the engineering challenges of deploying and maintaining Large Language Models in production. It focuses on scalability, reliability, and observability of AI systems.

## Structure

### 1. [Training Infrastructure](training-infrastructure/distributed/README.md)
Building scalable training pipelines.
- **Distributed Training**: FSDP, DeepSpeed, 3D Parallelism.
- **Data Pipelines**: High-throughput data processing.
- **Training Stability**: Handling loss spikes and hardware failures.
- **Hyperparameter Tuning**: Efficient search strategies.

### 2. [Model Serving](model-serving/architecture/README.md)
Efficient inference and deployment.
- **Serving Architecture**: vLLM, TGI, TensorRT-LLM.
- **Model Compression**: Quantization (AWQ, GPTQ), Pruning.
- **Model Registry**: Versioning and artifact management.

### 3. [Monitoring & Observability](monitoring/rag-observability/README.md)
Ensuring system health and quality.
- **RAG Observability**: Tracing retrieval and generation.
- **Drift Detection**: Monitoring data and concept drift.
- **Evaluation**: Automated testing and benchmarks.
- **Data Governance**: Privacy and compliance.

### 4. [Deployment Infrastructure](deployment/pipeline/README.md)
DevOps for AI.
- **CI/CD for ML**: Automated testing and deployment.
- **Cost Optimization**: Managing GPU resources and costs.
- **Security**: Reliability and threat protection.

---

**Next Phase**: [Capstone Projects](../07-Capstone-Projects/README.md)
