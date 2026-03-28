# LLM Frameworks and Tools Guide

**Documentation**: [**English**](README_EN.md) | [**中文**](README.md)

## Table of Contents

1. [Overview](#1-overview)
2. [HuggingFace Ecosystem](#2-huggingface-ecosystem)
3. [LangChain](#3-langchain)
4. [LlamaIndex](#4-llamaindex)
5. [vLLM and Inference Serving](#5-vllm-and-inference-serving)
6. [Other Tools](#6-other-tools)
7. [Framework Comparison](#7-framework-comparison)

---

## 1. Overview

The LLM ecosystem has evolved rapidly, producing a rich set of frameworks and tools. Choosing the right tool for each layer of the stack is critical for productivity and production readiness.

### 1.1 The LLM Tool Stack

```
┌─────────────────────────────────────────┐
│          Application Layer              │
│     (LangChain, LlamaIndex, Agents)    │
├─────────────────────────────────────────┤
│          Model Layer                    │
│  (Transformers, PEFT, TRL, Accelerate) │
├─────────────────────────────────────────┤
│          Serving Layer                  │
│       (vLLM, TGI, Ollama, TRT-LLM)    │
├─────────────────────────────────────────┤
│          Infrastructure Layer           │
│     (MLflow, W&B, Ray, DeepSpeed)      │
└─────────────────────────────────────────┘
```

---

## 2. HuggingFace Ecosystem

HuggingFace provides the foundational layer for most open-source LLM work.

### 2.1 Transformers

The core library for loading, using, and fine-tuning pre-trained models.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Key Features**:
- 200,000+ pre-trained models on the Hub
- Unified API across architectures (GPT, BERT, T5, LLaMA, etc.)
- Built-in training loop via `Trainer` class
- Pipeline API for quick inference

### 2.2 Datasets

Efficient data loading and processing for ML.

```python
from datasets import load_dataset

# Load from Hub
dataset = load_dataset("tatsu-lab/alpaca")

# Streaming for large datasets
dataset = load_dataset("cerebras/SlimPajama-627B", streaming=True)

# Custom preprocessing
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized = dataset.map(tokenize, batched=True)
```

**Key Features**:
- Arrow-based for memory-efficient processing
- Streaming mode for datasets larger than disk
- Built-in data versioning and caching

### 2.3 PEFT (Parameter-Efficient Fine-Tuning)

Fine-tune large models with minimal resources.

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

**Supported Methods**: LoRA, QLoRA, Prefix Tuning, Prompt Tuning, IA3

### 2.4 Accelerate

Distributed training without boilerplate.

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

**Key Features**:
- Single-GPU to multi-GPU/multi-node with zero code changes
- Mixed precision training (fp16, bf16)
- DeepSpeed and FSDP integration

### 2.5 TRL (Transformer Reinforcement Learning)

Fine-tune language models with reinforcement learning from human feedback.

```python
from trl import SFTTrainer, DPOTrainer

# Supervised Fine-Tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()

# DPO alignment
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_dataset,
    beta=0.1,
)
dpo_trainer.train()
```

---

## 3. LangChain

LangChain is the most popular framework for building LLM-powered applications.

### 3.1 Core Concepts

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Prompt  │ →  │  Model   │ →  │  Output  │
│ Template │    │  (LLM)   │    │  Parser  │
└──────────┘    └──────────┘    └──────────┘
         └──────────────┬──────────────┘
                     Chain
```

### 3.2 Chains (LCEL)

LangChain Expression Language for composing components.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | model | parser
result = chain.invoke({"topic": "quantum computing"})
```

### 3.3 Agents

LLMs that can use tools and make decisions.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun

tools = [DuckDuckGoSearchRun()]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What is the latest news about LLMs?"})
```

### 3.4 Memory

Maintain conversation state across interactions.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10)  # Keep last 10 exchanges

# Memory types:
# - ConversationBufferMemory: Store everything
# - ConversationBufferWindowMemory: Fixed window
# - ConversationSummaryMemory: LLM-summarized history
# - ConversationEntityMemory: Entity-based tracking
```

### 3.5 Retrieval (RAG)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain

# Build vector store
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Create RAG chain
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
result = rag_chain.invoke({"input": "How does attention work?"})
```

---

## 4. LlamaIndex

LlamaIndex specializes in connecting LLMs with data sources for knowledge-augmented generation.

### 4.1 Core Architecture

```
Data Sources → Data Connectors → Nodes → Index → Query Engine → Response
```

### 4.2 Data Connectors

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

# Load local files (PDF, DOCX, TXT, etc.)
documents = SimpleDirectoryReader("./data").load_data()

# Load from web
documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://example.com/docs"]
)
```

### 4.3 Index Types

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex

# Vector index — best for semantic search
vector_index = VectorStoreIndex.from_documents(documents)

# Summary index — best for summarization
summary_index = SummaryIndex.from_documents(documents)

# Tree index — best for hierarchical queries
tree_index = TreeIndex.from_documents(documents)
```

### 4.4 Query Engine

```python
# Basic query
query_engine = vector_index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What is transformer attention?")

# With response synthesis
from llama_index.core import get_response_synthesizer

synthesizer = get_response_synthesizer(response_mode="tree_summarize")
query_engine = vector_index.as_query_engine(
    response_synthesizer=synthesizer,
    similarity_top_k=10
)
```

### 4.5 LangChain vs LlamaIndex

| Feature | LangChain | LlamaIndex |
|---------|-----------|------------|
| **Focus** | General LLM app framework | Data-centric RAG |
| **Strength** | Agents, chains, tool use | Indexing, retrieval, data connectors |
| **Flexibility** | Higher — more components | Lower — opinionated data pipeline |
| **Learning Curve** | Steeper | Gentler for RAG use cases |
| **Best For** | Complex agent workflows | Document Q&A and knowledge bases |

---

## 5. vLLM and Inference Serving

### 5.1 vLLM

High-throughput LLM inference engine using PagedAttention.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["The future of AI is"], params)
print(outputs[0].outputs[0].text)
```

**OpenAI-compatible API server**:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Key Features**:
- PagedAttention for efficient KV-cache management
- Continuous batching for high throughput
- Tensor parallelism for multi-GPU serving
- 10-24x throughput vs naive HuggingFace inference

### 5.2 Text Generation Inference (TGI)

HuggingFace's production inference server.

```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf
```

### 5.3 Inference Framework Comparison

| Framework | Throughput | Latency | Ease of Use | Production Ready |
|-----------|-----------|---------|-------------|-----------------|
| **vLLM** | Very High | Low | Medium | Yes |
| **TGI** | High | Low | High (Docker) | Yes |
| **TRT-LLM** | Highest | Lowest | Low (complex setup) | Yes |
| **Ollama** | Medium | Medium | Very High | Development |
| **llama.cpp** | Medium | Low | Medium | Yes (CPU) |

---

## 6. Other Tools

### 6.1 Ollama

Run LLMs locally with a single command.

```bash
# Install and run
ollama run llama2

# API usage
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?"
}'
```

**Best For**: Local development, prototyping, privacy-sensitive applications.

### 6.2 MLflow

Experiment tracking and model management.

```python
import mlflow

mlflow.set_experiment("llm-finetuning")

with mlflow.start_run():
    mlflow.log_params({"learning_rate": 2e-5, "lora_r": 16})
    mlflow.log_metrics({"eval_loss": 0.45, "eval_accuracy": 0.82})
    mlflow.transformers.log_model(model, "model")
```

**Key Features**:
- Experiment tracking with metrics and artifacts
- Model registry for versioning and staging
- LLM evaluation with `mlflow.evaluate()`

### 6.3 Weights & Biases (W&B)

Comprehensive ML experiment tracking and visualization.

```python
import wandb

wandb.init(project="llm-training")
wandb.config.update({"model": "llama-2-7b", "lora_r": 16})

for step, loss in enumerate(training_losses):
    wandb.log({"loss": loss, "step": step})

wandb.finish()
```

**Key Features**:
- Real-time training dashboards
- Hyperparameter sweep
- Model and dataset versioning
- Team collaboration features

---

## 7. Framework Comparison

### 7.1 Decision Matrix

| Use Case | Recommended Stack |
|----------|------------------|
| **Quick prototyping** | Ollama + LangChain |
| **Document Q&A / RAG** | LlamaIndex + ChromaDB |
| **Complex agent workflows** | LangChain + OpenAI/Anthropic API |
| **Model fine-tuning** | HuggingFace (Transformers + PEFT + TRL) |
| **Production inference** | vLLM or TGI |
| **Experiment tracking** | MLflow or W&B |
| **Local/edge deployment** | Ollama or llama.cpp |

### 7.2 Full-Stack Example

A typical production LLM application stack:

```
Training:    HuggingFace Transformers + PEFT + Accelerate
Tracking:    W&B (training) + MLflow (model registry)
Serving:     vLLM (inference) + FastAPI (API layer)
Application: LangChain (orchestration) + ChromaDB (vector store)
Monitoring:  Prometheus + Grafana + custom eval pipeline
```

### 7.3 When to Use What

```
Need to fine-tune?
  ├── Yes → HuggingFace (Transformers + PEFT)
  └── No → Need custom app logic?
              ├── Yes → Complex agents? → LangChain
              │         RAG/data focus? → LlamaIndex
              └── No → Just need inference?
                        ├── Local dev → Ollama
                        └── Production → vLLM / TGI
```
