# Deep Learning & LLM Mastery

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Bilingual](https://img.shields.io/badge/Languages-EN%20%7C%20中文-blue.svg)]()

> **The Complete Engineering Guide: From ML Fundamentals to Production LLM Systems**

A professional, production-focused knowledge base designed for engineers, researchers, and technical leaders. This repository provides a structured path from classical machine learning to cutting-edge Large Language Model (LLM) engineering, Retrieval-Augmented Generation (RAG), and MLOps.

**🌐 Documentation**: [**English**](README_EN.md) | [**中文**](README.md)

---

## 🗺️ Learning Roadmap

The curriculum is organized into **7 Progressive Phases**, designed to build expertise layer by layer.

| Phase | Domain | Key Topics |
|-------|--------|------------|
| **01** | **Foundations** | Classical ML Algorithms, Deep Learning Basics |
| **02** | **Neural Networks** | CNNs, Sequence Models (RNN/LSTM), Optimization |
| **03** | **NLP & Transformers** | Attention Mechanisms, BERT, GPT, T5 Architecture |
| **04** | **LLM Core** | Pre-training, PEFT (LoRA), Alignment (RLHF/DPO), Prompt Engineering, Frameworks, Multimodal |
| **05** | **RAG & Agents** | Vector DBs, Advanced RAG, Agentic Patterns, Production Systems |
| **06** | **MLOps & Production** | Distributed Training, Serving, Monitoring, Benchmarks, Deployment Infrastructure |
| **07** | **Capstone Projects** | End-to-End Enterprise RAG & Fine-tuning Pipelines |

---

## 📂 Repository Structure

```
Daily-LLM/
│
├── 01-Foundations/               # 🟢 Phase 1: The Bedrock
│   ├── machine-learning/         # Algorithms, Math, Evaluation
│   └── deep-learning-basics/     # MLP, Backprop, Loss Functions
│
├── 02-Neural-Networks/           # 🟡 Phase 2: Deep Learning Patterns
│   ├── cnn-architectures/        # Computer Vision Architectures
│   ├── sequence-models/          # Sequence & Time-Series
│   └── training/                 # Modern Training Techniques
│
├── 03-NLP-Transformers/          # 🟠 Phase 3: The Transformer Revolution
│   ├── attention-mechanisms/     # Self-Attention Deep Dive
│   ├── transformer-architecture/ # Encoder-Decoder, Positional Encoding
│   └── pretrained-models/        # Model Families (BERT, GPT, T5)
│
├── 04-LLM-Core/                  # 🔴 Phase 4: Large Language Models
│   ├── pre-training/             # Data Pipelines, Scaling Laws
│   ├── peft/                     # Parameter-Efficient Fine-Tuning
│   ├── alignment/                # RLHF, DPO, Safety
│   ├── prompt-engineering/       # Prompt Design, CoT, Advanced Patterns
│   ├── frameworks/               # HuggingFace, LangChain, LlamaIndex, vLLM
│   └── multimodal/               # Vision-Language Models (CLIP, LLaVA)
│
├── 05-RAG-Systems/               # 🟣 Phase 5: RAG & Agents
│   ├── rag-foundations/          # Chunking, Embedding, Reranking
│   ├── vector-databases/         # Indexing, Retrieval
│   ├── agents/                   # ReAct, Planning, Tool Use
│   └── production/               # Industry Applications (Code, Search, etc.)
│
├── 06-MLOps-Production/          # 🔵 Phase 6: Engineering at Scale
│   ├── training-infrastructure/  # Distributed (FSDP/Deepspeed), Data Pipelines
│   ├── model-serving/            # vLLM, Optimization, Registry
│   ├── monitoring/               # Observability, Drift, Evaluation, Benchmarks
│   └── deployment/               # K8s, CI/CD, Cost Optimization
│
└── 07-Capstone-Projects/         # ⚫ Phase 7: Real-World Implementation
    ├── enterprise-rag-system/    # Production RAG with Agents
    └── finetune-deploy-pipeline/ # Automated Fine-tuning Pipeline
```

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Hardware**: CUDA-capable GPU recommended for LLM phases.

### Installation

```bash
git clone https://github.com/zkywsg/Daily-LLM.git
cd Daily-LLM

# Install all dependencies
pip install -r requirements.txt

# Or install by learning phase:
# Phase 1-2: pip install torch numpy scikit-learn matplotlib
# Phase 3-4: pip install transformers datasets peft trl sentence-transformers
# Phase 5:   pip install sentence-transformers faiss-cpu chromadb langchain
# Phase 6-7: pip install vllm fastapi mlflow wandb
```

---

## 🎯 Target Audience

- **ML Engineers**: Transitioning from classical ML to LLMs.
- **Software Engineers**: Building AI-powered applications (RAG/Agents).
- **Researchers**: Understanding the "Why" behind the "How".
- **Technical Leaders**: Designing scalable AI infrastructure.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
