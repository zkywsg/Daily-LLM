# 深度学习与大模型精通之路 (Deep Learning & LLM Mastery)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Bilingual](https://img.shields.io/badge/Languages-EN%20%7C%20中文-blue.svg)]()

> **全栈工程指南：从机器学习基础到生产级大模型系统**

这是一个为工程师、研究人员和技术领导者打造的专业级、面向生产的知识库。本仓库提供了一条结构化的学习路径，涵盖从经典机器学习到前沿大模型 (LLM) 工程、检索增强生成 (RAG) 和 MLOps 的完整技术栈。

**🌐 文档语言**: [**English**](README.md) | [**中文**](README_CN.md)

---

## 🗺️ 学习路线图

课程体系划分为 **7 个渐进阶段**，旨在层层递进地构建核心能力。

| 阶段 | 领域 | 核心主题 |
|------|------|----------|
| **01** | **基础篇 (Foundations)** | 经典 ML 算法、深度学习基础数学 |
| **02** | **神经网络 (Neural Networks)** | CNN、序列模型 (RNN/LSTM)、优化技术 |
| **03** | **NLP 与 Transformer** | 注意力机制、BERT、GPT、T5 架构 |
| **04** | **大模型核心 (LLM Core)** | 预训练、高效微调 (PEFT)、对齐 (RLHF/DPO)、多模态 |
| **05** | **RAG 与 Agent** | 向量数据库、高级 RAG、Agent 模式、生产级应用 |
| **06** | **MLOps 与生产工程** | 分布式训练、模型服务、监控观测、基础设施部署 |
| **07** | **实战项目 (Capstone)** | 端到端企业级 RAG 与微调流水线 |

---

## 📂 知识库结构

```
Daily-LLM/
│
├── 01-Foundations/               # 🟢 Phase 1: 基石
│   ├── machine-learning/         # 算法原理、数学基础、评估指标
│   ├── deep-learning-basics/     # 多层感知机、反向传播、损失函数
│
├── 02-Neural-Networks/           # 🟡 Phase 2: 深度学习模式
│   ├── cnn-architectures/        # 计算机视觉架构
│   ├── sequence-models/          # 序列与时间序列处理
│   ├── training/                 # 现代训练技巧与优化
│
├── 03-NLP-Transformers/          # 🟠 Phase 3: Transformer 革命
│   ├── attention-mechanisms/     # 自注意力机制深度解析
│   ├── transformer-architecture/ # 编码器-解码器架构
│   ├── pretrained-models/        # 预训练模型家族 (BERT, GPT, T5)
│
├── 04-LLM-Core/                  # 🔴 Phase 4: 大语言模型
│   ├── pre-training/             # 数据流水线、Scaling Laws
│   ├── peft/                     # 参数高效微调 (LoRA/QLoRA)
│   ├── alignment/                # 对齐技术 (RLHF, DPO, 安全性)
│   ├── multimodal/               # 视觉语言模型 (CLIP, LLaVA)
│
├── 05-RAG-Systems/               # 🟣 Phase 5: RAG 与 智能体
│   ├── rag-foundations/          # 分块、Embedding、重排序 (Rerank)
│   ├── vector-databases/         # 索引与检索技术
│   ├── agents/                   # ReAct、规划、工具调用
│   ├── production/               # 行业应用场景 (代码助手、搜索等)
│
├── 06-MLOps-Production/          # 🔵 Phase 6: 大规模工程化
│   ├── training-infrastructure/  # 分布式训练 (FSDP/Deepspeed)
│   ├── model-serving/            # 推理服务 (vLLM)、优化、模型仓库
│   ├── monitoring/               # 可观测性、漂移检测、评估
│   ├── deployment/               # K8s、CI/CD、成本优化
│
└── 07-Capstone-Projects/         # ⚫ Phase 7: 实战落地
    ├── enterprise-rag-system/    # 生产级 RAG + Agent 系统
    ├── finetune-deploy-pipeline/ # 自动化微调与部署流水线
```

---

## 🚀 快速开始

### 前置要求
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **硬件**: 学习 LLM 相关阶段建议配备支持 CUDA 的 GPU。

### 安装

```bash
git clone https://github.com/zkywsg/Daily-LLM.git
cd Daily-LLM

# 安装核心依赖
pip install -r requirements.txt
```

---

## 🎯 目标受众

- **机器学习工程师**: 从传统 ML 向 LLM/GenAI 转型。
- **软件工程师**: 构建基于 AI 的应用 (RAG/Agents)。
- **研究人员**: 深入理解技术背后的原理。
- **技术负责人**: 设计可扩展的 AI 基础设施。

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解我们的行为准则和提交 Pull Request 的流程。

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
