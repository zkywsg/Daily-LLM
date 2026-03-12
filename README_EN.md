<div align="center">

<img src=".github/assets/readme-banner.svg" alt="Daily-LLM banner" width="100%" />

# Deep Learning & LLM Mastery

### A complete engineering roadmap from ML fundamentals to production-grade LLM systems

<p>
  Built for engineers, researchers, and technical leaders who want more than scattered notes.<br/>
  This is a bilingual knowledge base focused on architecture, implementation, evaluation, and real-world deployment.
</p>

<p>
  <a href="README.md"><strong>中文</strong></a>
  ·
  <a href="#highlights"><strong>Highlights</strong></a>
  ·
  <a href="#why-this-repo"><strong>Why This Repo</strong></a>
  ·
  <a href="#learning-map"><strong>Learning Map</strong></a>
  ·
  <a href="#quick-start"><strong>Quick Start</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-0F172A.svg?style=flat-square)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-2563EB.svg?style=flat-square)](https://www.python.org/downloads/)
[![Phases](https://img.shields.io/badge/Phases-7-059669.svg?style=flat-square)](./)
[![Modules](https://img.shields.io/badge/Core%20Modules-24-7C3AED.svg?style=flat-square)](./)
[![Bilingual](https://img.shields.io/badge/Docs-English%20%7C%20%E4%B8%AD%E6%96%87-D97706.svg?style=flat-square)](README.md)

</div>

> [!TIP]
> If this repository saves you research time or helps you build faster, consider leaving a star.

<a id="highlights"></a>

## Highlights

<table>
  <tr>
    <td width="25%" align="center" valign="top">
      <strong><sub>PHASES</sub></strong><br/>
      <strong>7</strong><br/>
      Progressive stages
    </td>
    <td width="25%" align="center" valign="top">
      <strong><sub>MODULES</sub></strong><br/>
      <strong>24</strong><br/>
      Core first-level modules
    </td>
    <td width="25%" align="center" valign="top">
      <strong><sub>COVERAGE</sub></strong><br/>
      <strong>ML → LLM → RAG → MLOps</strong><br/>
      From fundamentals to production
    </td>
    <td width="25%" align="center" valign="top">
      <strong><sub>DOCS</sub></strong><br/>
      <strong>English + 中文</strong><br/>
      Built for long-term reference
    </td>
  </tr>
</table>

> One continuous engineering roadmap connecting principles, architecture, implementation, evaluation, and deployment.

<a id="why-this-repo"></a>

## Why This Repo

<table>
  <tr>
    <td width="33%" valign="top">
      <strong>Structured, not fragmented</strong><br/>
      The roadmap moves from classical ML to neural networks, Transformers, LLMs, RAG, agents, MLOps, and capstone systems in one connected sequence.
    </td>
    <td width="33%" valign="top">
      <strong>Engineering-first, not theory-only</strong><br/>
      The content emphasizes system design, implementation tradeoffs, evaluation, serving, observability, deployment, and production concerns.
    </td>
    <td width="33%" valign="top">
      <strong>Bilingual and built to revisit</strong><br/>
      Core documentation is available in both English and Chinese, making it useful for learning, team sharing, and long-term reference.
    </td>
  </tr>
</table>

## At A Glance

| Dimension | Summary |
| --- | --- |
| Learning path | 7 progressive phases from fundamentals to production |
| Core modules | 24 first-level modules organized by topic |
| Focus | Principles + engineering + architecture + practical implementation |
| Audience | ML engineers, software engineers, researchers, technical leaders |
| Language | English / 中文 |

## What You Get From This Repository

- A coherent path from classical machine learning to Transformers, LLMs, RAG, agents, and production systems.
- A stronger intuition for when to use which techniques, frameworks, and architectures.
- Practical building blocks for fine-tuning, retrieval, serving, evaluation, observability, and deployment.
- A knowledge base that is easier to return to than a pile of disconnected articles and videos.

<a id="learning-map"></a>

## Learning Map

| Phase | Focus | What You Will Learn | Entry |
| --- | --- | --- | --- |
| **01** | Foundations | Classical ML, math basics, evaluation, deep learning fundamentals | [Open Phase 01](01-Foundations/) |
| **02** | Neural Networks | CNNs, sequence models, optimization, modern training patterns | [Open Phase 02](02-Neural-Networks/) |
| **03** | NLP & Transformers | Attention, Transformer architecture, BERT/GPT/T5 families | [Open Phase 03](03-NLP-Transformers/) |
| **04** | LLM Core | Pre-training, PEFT, alignment, prompt engineering, frameworks, multimodal | [Open Phase 04](04-LLM-Core/) |
| **05** | RAG & Agents | Retrieval systems, vector databases, tool use, multi-agent patterns | [Open Phase 05](05-RAG-Systems/README_EN.md) |
| **06** | MLOps & Production | Training infra, serving, monitoring, benchmarking, deployment | [Open Phase 06](06-MLOps-Production/README_EN.md) |
| **07** | Capstone Projects | Enterprise-grade RAG and fine-tuning/deployment pipelines | [Open Phase 07](07-Capstone-Projects/) |

## Pick A Starting Path

| Your goal | Recommended sequence |
| --- | --- |
| Build from fundamentals | `01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07` |
| Ship RAG / agent systems fast | `03 -> 04 -> 05 -> 06 -> 07` |
| Focus on training and fine-tuning | `02 -> 03 -> 04 -> 06 -> 07` |
| Lead AI architecture and delivery | `04 -> 05 -> 06 -> 07` |

<a id="structure"></a>

## Repository Structure

<details>
<summary><strong>Expand directory overview</strong></summary>

```text
Daily-LLM/
├── 01-Foundations/          # ML and deep learning fundamentals
├── 02-Neural-Networks/      # CNNs, sequence models, training and optimization
├── 03-NLP-Transformers/     # Attention and Transformer systems
├── 04-LLM-Core/             # Pre-training, PEFT, alignment, prompting, multimodal
├── 05-RAG-Systems/          # RAG, vector retrieval, agents, production patterns
├── 06-MLOps-Production/     # Infra, serving, monitoring, deployment
├── 07-Capstone-Projects/    # End-to-end enterprise projects
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md
└── README_EN.md
```

</details>

## Typical Topics Covered Here

- **Model fundamentals**: algorithms, backpropagation, optimization, evaluation metrics.
- **Transformer systems**: self-attention, encoder-decoder design, pretrained model families.
- **LLM engineering**: pre-training pipelines, LoRA/QLoRA, RLHF, DPO, prompting, multimodal stacks.
- **RAG and agents**: chunking, embeddings, reranking, query rewriting, tool use, memory, multi-agent systems.
- **Production systems**: vLLM, model serving, observability, benchmarks, CI/CD, Kubernetes, cost optimization.
- **Capstone applications**: enterprise search, code assistants, conversational systems, automated fine-tuning pipelines.

<a id="quick-start"></a>

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/zkywsg/Daily-LLM.git
cd Daily-LLM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>Install by phase</strong></summary>

```bash
# Phase 1-2
pip install torch numpy scikit-learn matplotlib

# Phase 3-4
pip install transformers datasets peft trl sentence-transformers

# Phase 5
pip install sentence-transformers faiss-cpu chromadb langchain

# Phase 6-7
pip install vllm fastapi mlflow wandb
```

</details>

### 3. Start where it matters most to you

- Want fundamentals first: start with [01-Foundations](01-Foundations/).
- Want to move directly into LLM engineering: start with [04-LLM-Core](04-LLM-Core/).
- Want retrieval and agent systems: start with [05-RAG-Systems](05-RAG-Systems/README_EN.md).
- Want production delivery and operations: start with [06-MLOps-Production](06-MLOps-Production/README_EN.md).

## Who This Is For

- **ML Engineers**: moving from traditional ML into LLM and GenAI systems.
- **Software Engineers**: building AI products, not just calling an API.
- **Researchers**: connecting intuition, architecture, and implementation detail.
- **Technical Leaders**: designing scalable, observable, maintainable AI platforms.

## Contributing

Contributions are welcome. If you want to improve content, add examples, or refine the structure, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
  If this repository is useful, a star helps prioritize future curation and expansion.
</div>
