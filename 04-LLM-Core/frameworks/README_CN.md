# LLM 框架与工具指南

**文档语言**: [**English**](README.md) | [**中文**](README_CN.md)

## 目录

1. [概述](#1-概述)
2. [HuggingFace 生态](#2-huggingface-生态)
3. [LangChain](#3-langchain)
4. [LlamaIndex](#4-llamaindex)
5. [vLLM 与推理服务](#5-vllm-与推理服务)
6. [其他工具](#6-其他工具)
7. [框架选型对比](#7-框架选型对比)

---

## 1. 概述

LLM 生态系统快速发展，产生了丰富的框架和工具。为每一层选择合适的工具对于生产力和生产就绪性至关重要。

### 1.1 LLM 工具栈

```
┌─────────────────────────────────────────┐
│           应用层                         │
│     (LangChain, LlamaIndex, Agents)    │
├─────────────────────────────────────────┤
│           模型层                         │
│  (Transformers, PEFT, TRL, Accelerate) │
├─────────────────────────────────────────┤
│           服务层                         │
│       (vLLM, TGI, Ollama, TRT-LLM)    │
├─────────────────────────────────────────┤
│           基础设施层                     │
│     (MLflow, W&B, Ray, DeepSpeed)      │
└─────────────────────────────────────────┘
```

---

## 2. HuggingFace 生态

HuggingFace 为大多数开源 LLM 工作提供基础层。

### 2.1 Transformers

加载、使用和微调预训练模型的核心库。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

inputs = tokenizer("人工智能的未来是", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**核心特性**：
- Hub 上 200,000+ 预训练模型
- 跨架构统一 API（GPT、BERT、T5、LLaMA 等）
- 通过 `Trainer` 类内置训练循环
- Pipeline API 实现快速推理

### 2.2 Datasets

高效的 ML 数据加载和处理。

```python
from datasets import load_dataset

# 从 Hub 加载
dataset = load_dataset("tatsu-lab/alpaca")

# 流式加载大型数据集
dataset = load_dataset("cerebras/SlimPajama-627B", streaming=True)

# 自定义预处理
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized = dataset.map(tokenize, batched=True)
```

**核心特性**：
- 基于 Arrow 格式，内存高效
- 支持流式模式处理超大数据集
- 内置数据版本控制和缓存

### 2.3 PEFT（参数高效微调）

以最小资源微调大型模型。

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
# 可训练参数: 4,194,304 || 总参数: 6,742,609,920 || 可训练比例: 0.0622%
```

**支持方法**：LoRA、QLoRA、Prefix Tuning、Prompt Tuning、IA3

### 2.4 Accelerate

无需样板代码的分布式训练。

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

**核心特性**：
- 单卡到多卡/多节点零代码改动
- 混合精度训练（fp16, bf16）
- DeepSpeed 和 FSDP 集成

### 2.5 TRL（Transformer 强化学习）

使用人类反馈强化学习微调语言模型。

```python
from trl import SFTTrainer, DPOTrainer

# 监督微调
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()

# DPO 对齐
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

LangChain 是构建 LLM 驱动应用最流行的框架。

### 3.1 核心概念

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Prompt  │ →  │  模型    │ →  │  输出    │
│  模板    │    │  (LLM)   │    │  解析器  │
└──────────┘    └──────────┘    └──────────┘
         └──────────────┬──────────────┘
                      Chain
```

### 3.2 Chains（LCEL）

LangChain 表达式语言，用于组合组件。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用简单的话解释{topic}。")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# 使用管道操作符组合
chain = prompt | model | parser
result = chain.invoke({"topic": "量子计算"})
```

### 3.3 Agents

能使用工具并做决策的 LLM。

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun

tools = [DuckDuckGoSearchRun()]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "LLM 的最新新闻是什么？"})
```

### 3.4 Memory（记忆）

跨交互维护对话状态。

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10)  # 保留最近 10 轮对话

# 记忆类型：
# - ConversationBufferMemory: 存储所有内容
# - ConversationBufferWindowMemory: 固定窗口
# - ConversationSummaryMemory: LLM 总结历史
# - ConversationEntityMemory: 基于实体的跟踪
```

### 3.5 检索（RAG）

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain

# 构建向量存储
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 创建 RAG 链
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
result = rag_chain.invoke({"input": "注意力机制是如何工作的？"})
```

---

## 4. LlamaIndex

LlamaIndex 专注于连接 LLM 与数据源，实现知识增强生成。

### 4.1 核心架构

```
数据源 → 数据连接器 → 节点 → 索引 → 查询引擎 → 响应
```

### 4.2 数据连接器

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

# 加载本地文件（PDF、DOCX、TXT 等）
documents = SimpleDirectoryReader("./data").load_data()

# 从网页加载
documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://example.com/docs"]
)
```

### 4.3 索引类型

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex

# 向量索引 — 最适合语义搜索
vector_index = VectorStoreIndex.from_documents(documents)

# 摘要索引 — 最适合总结
summary_index = SummaryIndex.from_documents(documents)

# 树索引 — 最适合分层查询
tree_index = TreeIndex.from_documents(documents)
```

### 4.4 查询引擎

```python
# 基本查询
query_engine = vector_index.as_query_engine(similarity_top_k=5)
response = query_engine.query("什么是 Transformer 注意力机制？")

# 带响应合成
from llama_index.core import get_response_synthesizer

synthesizer = get_response_synthesizer(response_mode="tree_summarize")
query_engine = vector_index.as_query_engine(
    response_synthesizer=synthesizer,
    similarity_top_k=10
)
```

### 4.5 LangChain vs LlamaIndex

| 特性 | LangChain | LlamaIndex |
|------|-----------|------------|
| **聚焦** | 通用 LLM 应用框架 | 以数据为中心的 RAG |
| **优势** | Agent、链、工具调用 | 索引、检索、数据连接器 |
| **灵活性** | 更高——更多组件 | 较低——有主张的数据流水线 |
| **学习曲线** | 较陡 | RAG 用例更平缓 |
| **最适场景** | 复杂 Agent 工作流 | 文档问答和知识库 |

---

## 5. vLLM 与推理服务

### 5.1 vLLM

使用 PagedAttention 的高吞吐 LLM 推理引擎。

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["人工智能的未来是"], params)
print(outputs[0].outputs[0].text)
```

**兼容 OpenAI 的 API 服务器**：

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
    messages=[{"role": "user", "content": "你好！"}]
)
```

**核心特性**：
- PagedAttention 实现高效 KV-cache 管理
- 连续批处理实现高吞吐
- 张量并行支持多 GPU 服务
- 相比原生 HuggingFace 推理提升 10-24x 吞吐量

### 5.2 Text Generation Inference (TGI)

HuggingFace 的生产级推理服务器。

```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf
```

### 5.3 推理框架对比

| 框架 | 吞吐量 | 延迟 | 易用性 | 生产就绪 |
|------|--------|------|--------|---------|
| **vLLM** | 极高 | 低 | 中等 | 是 |
| **TGI** | 高 | 低 | 高（Docker） | 是 |
| **TRT-LLM** | 最高 | 最低 | 低（配置复杂） | 是 |
| **Ollama** | 中等 | 中等 | 极高 | 开发环境 |
| **llama.cpp** | 中等 | 低 | 中等 | 是（CPU） |

---

## 6. 其他工具

### 6.1 Ollama

一键本地运行 LLM。

```bash
# 安装并运行
ollama run llama2

# API 调用
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "为什么天空是蓝色的？"
}'
```

**最适场景**：本地开发、原型验证、隐私敏感场景。

### 6.2 MLflow

实验追踪和模型管理。

```python
import mlflow

mlflow.set_experiment("llm-finetuning")

with mlflow.start_run():
    mlflow.log_params({"learning_rate": 2e-5, "lora_r": 16})
    mlflow.log_metrics({"eval_loss": 0.45, "eval_accuracy": 0.82})
    mlflow.transformers.log_model(model, "model")
```

**核心特性**：
- 带指标和产物的实验追踪
- 模型注册表支持版本管理和阶段管理
- 通过 `mlflow.evaluate()` 进行 LLM 评估

### 6.3 Weights & Biases (W&B)

全面的 ML 实验追踪和可视化。

```python
import wandb

wandb.init(project="llm-training")
wandb.config.update({"model": "llama-2-7b", "lora_r": 16})

for step, loss in enumerate(training_losses):
    wandb.log({"loss": loss, "step": step})

wandb.finish()
```

**核心特性**：
- 实时训练仪表盘
- 超参数搜索
- 模型和数据集版本管理
- 团队协作功能

---

## 7. 框架选型对比

### 7.1 决策矩阵

| 用例 | 推荐技术栈 |
|------|-----------|
| **快速原型验证** | Ollama + LangChain |
| **文档问答 / RAG** | LlamaIndex + ChromaDB |
| **复杂 Agent 工作流** | LangChain + OpenAI/Anthropic API |
| **模型微调** | HuggingFace (Transformers + PEFT + TRL) |
| **生产推理** | vLLM 或 TGI |
| **实验追踪** | MLflow 或 W&B |
| **本地/边缘部署** | Ollama 或 llama.cpp |

### 7.2 全栈示例

典型的生产 LLM 应用技术栈：

```
训练:     HuggingFace Transformers + PEFT + Accelerate
追踪:     W&B（训练）+ MLflow（模型注册）
服务:     vLLM（推理）+ FastAPI（API 层）
应用:     LangChain（编排）+ ChromaDB（向量存储）
监控:     Prometheus + Grafana + 自定义评估流水线
```

### 7.3 什么时候用什么

```
需要微调？
  ├── 是 → HuggingFace (Transformers + PEFT)
  └── 否 → 需要自定义应用逻辑？
              ├── 是 → 复杂 Agent？ → LangChain
              │         RAG/数据聚焦？ → LlamaIndex
              └── 否 → 只需要推理？
                        ├── 本地开发 → Ollama
                        └── 生产部署 → vLLM / TGI
```
