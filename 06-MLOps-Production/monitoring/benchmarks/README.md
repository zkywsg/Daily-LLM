# LLM Evaluation Benchmarks

**Documentation**: [**English**](README.md) | [**中文**](README_CN.md)

## Table of Contents

1. [Overview](#1-overview)
2. [General Knowledge Benchmarks](#2-general-knowledge-benchmarks)
3. [Code Benchmarks](#3-code-benchmarks)
4. [Reasoning Benchmarks](#4-reasoning-benchmarks)
5. [Safety Benchmarks](#5-safety-benchmarks)
6. [Leaderboards](#6-leaderboards)
7. [Practical Guide — lm-eval-harness](#7-practical-guide--lm-eval-harness)
8. [Best Practices](#8-best-practices)

---

## 1. Overview

### 1.1 Why Benchmarks Matter

Benchmarks provide standardized, reproducible ways to measure and compare LLM capabilities. They answer critical questions:
- How does Model A compare to Model B on reasoning tasks?
- Has my fine-tuned model improved or regressed?
- Is this model safe to deploy in production?

### 1.2 Benchmark Taxonomy

```
LLM Benchmarks
├── General Knowledge ── MMLU, HellaSwag, ARC, TruthfulQA, Winogrande
├── Code Generation  ── HumanEval, MBPP, SWE-bench
├── Reasoning        ── GSM8K, MATH, BBH
├── Safety           ── ToxiGen, RealToxicityPrompts
└── Domain-Specific  ── MedQA, LegalBench, FinBench
```

### 1.3 Key Terminology

| Term | Definition |
|------|-----------|
| **N-shot** | Number of examples provided in the prompt (0-shot, 5-shot, etc.) |
| **Pass@k** | Probability of at least one correct solution in k attempts |
| **Contamination** | Benchmark data leaking into training data |
| **Saturation** | Models reaching near-perfect scores, reducing benchmark utility |

---

## 2. General Knowledge Benchmarks

### 2.1 MMLU (Massive Multitask Language Understanding)

- **What**: 57 subjects covering STEM, humanities, social sciences, and more
- **Format**: Multiple-choice (4 options), 14,042 questions
- **Evaluation**: 5-shot accuracy
- **Significance**: The most widely cited general knowledge benchmark

| Model | MMLU Score |
|-------|-----------|
| GPT-4 | 86.4% |
| Claude 3.5 Sonnet | 88.7% |
| Llama 3 70B | 82.0% |
| Mixtral 8x7B | 70.6% |

### 2.2 HellaSwag

- **What**: Sentence completion requiring commonsense reasoning
- **Format**: 4-way multiple choice, 10,042 questions
- **Evaluation**: 10-shot accuracy
- **Key Insight**: Tests grounded commonsense, not just factual recall

### 2.3 ARC (AI2 Reasoning Challenge)

- **What**: Grade-school science questions
- **Format**: Multiple-choice; split into Easy (5,197) and Challenge (2,590) sets
- **Evaluation**: 25-shot accuracy on Challenge set
- **Key Insight**: Challenge set filters out questions answerable by simple retrieval

### 2.4 TruthfulQA

- **What**: Tests whether models generate truthful answers vs. common misconceptions
- **Format**: 817 questions across 38 categories
- **Evaluation**: Human-evaluated truthfulness + informativeness
- **Key Insight**: Larger models can score *worse* due to better memorization of misconceptions

### 2.5 Winogrande

- **What**: Pronoun resolution requiring commonsense reasoning
- **Format**: Binary choice, 44,000 problems
- **Evaluation**: 5-shot accuracy
- **Example**: "The trophy doesn't fit in the suitcase because *it* is too [big/small]."

---

## 3. Code Benchmarks

### 3.1 HumanEval

- **What**: 164 hand-crafted Python programming problems
- **Format**: Function signature + docstring → complete implementation
- **Evaluation**: Pass@1, Pass@10, Pass@100 (unit test pass rate)
- **Created by**: OpenAI

```python
# Example HumanEval problem
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers in the list are closer than the given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
    True
    """
```

| Model | Pass@1 |
|-------|--------|
| GPT-4 | 67.0% |
| Claude 3.5 Sonnet | 92.0% |
| Code Llama 34B | 48.8% |

### 3.2 MBPP (Mostly Basic Python Problems)

- **What**: 974 crowd-sourced Python programming tasks
- **Format**: Task description + 3 test cases → solution
- **Evaluation**: Pass@1
- **Key Difference from HumanEval**: Simpler problems, larger dataset

### 3.3 SWE-bench

- **What**: Real-world software engineering tasks from GitHub issues
- **Format**: Given a codebase + issue description → generate a patch
- **Evaluation**: Percentage of issues resolved (verified by existing test suites)
- **Key Insight**: Tests real-world coding ability, not just isolated function generation

---

## 4. Reasoning Benchmarks

### 4.1 GSM8K (Grade School Math 8K)

- **What**: 8,500 grade-school math word problems
- **Format**: Natural language problem → step-by-step solution + numeric answer
- **Evaluation**: Exact match on final answer
- **Key Insight**: Tests multi-step arithmetic reasoning; benefits strongly from Chain-of-Thought

| Model | GSM8K (CoT) |
|-------|------------|
| GPT-4 | 92.0% |
| Llama 3 70B | 83.7% |
| Mistral 7B | 52.2% |

### 4.2 MATH

- **What**: 12,500 competition-level math problems (AMC, AIME, etc.)
- **Format**: Problem → LaTeX solution
- **Evaluation**: Exact match on final answer
- **Difficulty Levels**: 1-5 (Level 5 = competition difficulty)
- **Key Insight**: Much harder than GSM8K; tests genuine mathematical reasoning

### 4.3 BBH (BIG-Bench Hard)

- **What**: 23 challenging tasks from BIG-Bench where LLMs previously underperformed
- **Format**: Task-specific (logical deduction, date understanding, causal judgment, etc.)
- **Evaluation**: 3-shot CoT accuracy
- **Key Insight**: Specifically curated to be tasks where scaling alone doesn't help

---

## 5. Safety Benchmarks

### 5.1 ToxiGen

- **What**: Machine-generated toxic and benign statements targeting 13 minority groups
- **Format**: 274,186 statements → classify as toxic or benign
- **Evaluation**: Toxicity classification accuracy
- **Key Insight**: Tests both toxicity detection and generation bias

### 5.2 RealToxicityPrompts

- **What**: 100,000 naturally occurring prompts from web text
- **Format**: Prompt → measure toxicity of model continuation
- **Evaluation**: Expected maximum toxicity, toxicity probability
- **Key Insight**: Tests how likely models are to generate toxic content given benign-looking prompts

### 5.3 Other Safety Evaluations

| Benchmark | Focus Area |
|-----------|-----------|
| **BBQ** | Social bias in question answering |
| **WinoBias** | Gender bias in coreference resolution |
| **CrowS-Pairs** | Stereotypical associations |
| **HarmBench** | Adversarial attack robustness |

---

## 6. Leaderboards

### 6.1 Open LLM Leaderboard (Hugging Face)

- **URL**: huggingface.co/spaces/open-llm-leaderboard
- **Benchmarks**: MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K
- **Scope**: Open-weight models only
- **Updates**: Continuous community submissions
- **Value**: The go-to reference for comparing open-source models

### 6.2 Chatbot Arena (LMSYS)

- **URL**: chat.lmsys.org
- **Method**: Blind A/B comparisons by human voters
- **Metric**: Elo rating system (like chess)
- **Scope**: Both open and closed models
- **Value**: Most reflective of real-world human preferences

### 6.3 MTEB (Massive Text Embedding Benchmark)

- **URL**: huggingface.co/spaces/mteb/leaderboard
- **Focus**: Embedding model evaluation
- **Tasks**: Retrieval, classification, clustering, STS, reranking
- **Value**: The standard for comparing embedding models (relevant for RAG)

### 6.4 Leaderboard Comparison

| Leaderboard | Evaluation Type | Models Covered | Best For |
|-------------|----------------|---------------|----------|
| Open LLM Leaderboard | Automated benchmarks | Open-weight | Objective comparison |
| Chatbot Arena | Human preference | Open + closed | Real-world quality |
| MTEB | Automated (embedding) | Embedding models | RAG/search selection |
| AlpacaEval | LLM-as-judge | Open + closed | Instruction following |

---

## 7. Practical Guide — lm-eval-harness

### 7.1 Overview

[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the standard open-source framework for evaluating language models. It supports 200+ benchmarks and is used by the Hugging Face Open LLM Leaderboard.

### 7.2 Installation

```bash
pip install lm-eval
```

### 7.3 Basic Usage

```bash
# Evaluate a HuggingFace model on MMLU
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 8 \
    --output_path ./results/

# Evaluate on multiple benchmarks
lm_eval --model hf \
    --model_args pretrained=mistralai/Mistral-7B-v0.1 \
    --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,gsm8k \
    --batch_size auto \
    --output_path ./results/
```

### 7.4 Evaluate OpenAI Models

```bash
# Using the OpenAI API
lm_eval --model openai-completions \
    --model_args model=gpt-3.5-turbo \
    --tasks mmlu \
    --num_fewshot 5
```

### 7.5 Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["mmlu", "gsm8k"],
    num_fewshot=5,
    batch_size=8,
)

# Print results
for task, metrics in results["results"].items():
    print(f"{task}: {metrics['acc,none']:.4f}")
```

### 7.6 Custom Evaluation Tasks

```yaml
# my_custom_task.yaml
task: my_custom_qa
dataset_path: my_org/my_dataset
dataset_name: default
output_type: generate_until
training_split: train
test_split: test
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
```

```bash
# Run custom task
lm_eval --model hf \
    --model_args pretrained=my-model \
    --tasks my_custom_qa \
    --include_path ./custom_tasks/
```

---

## 8. Best Practices

### 8.1 Benchmark Contamination

**Problem**: If benchmark data appears in the training set, scores are inflated and unreliable.

**Detection Methods**:
- N-gram overlap analysis between training data and benchmark
- Canary string insertion in test sets
- Perplexity analysis on held-out vs. benchmark data

**Mitigation**:
- Use held-out test sets not publicly available
- Regularly create new evaluation sets
- Report training data provenance alongside scores

### 8.2 Reporting Standards

When reporting benchmark results, always include:

| Required | Description |
|----------|-------------|
| Model version | Exact model identifier and checkpoint |
| N-shot setting | Number of few-shot examples used |
| Prompt format | Exact template used for evaluation |
| Evaluation framework | Tool and version (e.g., lm-eval-harness v0.4.0) |
| Quantization | If applicable (e.g., 4-bit, 8-bit) |
| Hardware | GPU type and count |

### 8.3 Custom Evaluation Design

When standard benchmarks don't cover your use case:

1. **Define Clear Success Criteria**: What does "good" look like for your application?
2. **Build Domain-Specific Test Sets**: 100-500 representative examples with gold labels
3. **Use Multiple Evaluation Methods**:
   - Automated metrics (exact match, F1, BLEU, ROUGE)
   - LLM-as-judge (GPT-4 / Claude scoring)
   - Human evaluation (for subjective quality)
4. **Version Your Test Sets**: Track changes to prevent accidental contamination
5. **Report Confidence Intervals**: Use bootstrap sampling for statistical significance

### 8.4 Common Pitfalls

1. **Cherry-picking Benchmarks**: Only reporting benchmarks where your model excels
2. **Ignoring Variance**: Not reporting standard deviation across runs
3. **Prompt Sensitivity**: Results can vary 5-15% with different prompt templates
4. **Benchmark Saturation**: Models scoring 95%+ make the benchmark less useful for differentiation
5. **Over-reliance on Leaderboards**: A model ranking #1 on MMLU may still fail at your specific task
6. **Neglecting Latency/Cost**: Benchmark scores don't capture inference speed or cost efficiency
