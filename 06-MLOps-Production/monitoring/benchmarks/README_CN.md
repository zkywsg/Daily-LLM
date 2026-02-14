# LLM 评测基准

**文档语言**: [**English**](README.md) | [**中文**](README_CN.md)

## 目录

1. [概述](#1-概述)
2. [通用知识基准](#2-通用知识基准)
3. [代码基准](#3-代码基准)
4. [推理基准](#4-推理基准)
5. [安全基准](#5-安全基准)
6. [排行榜](#6-排行榜)
7. [实操指南 — lm-eval-harness](#7-实操指南--lm-eval-harness)
8. [最佳实践](#8-最佳实践)

---

## 1. 概述

### 1.1 为什么基准测试很重要？

基准测试提供了标准化、可复现的方式来衡量和比较 LLM 能力。它们回答了关键问题：
- 模型 A 和模型 B 在推理任务上如何比较？
- 我微调后的模型是提升了还是退化了？
- 这个模型是否安全到可以在生产中部署？

### 1.2 基准分类

```
LLM 基准
├── 通用知识 ── MMLU, HellaSwag, ARC, TruthfulQA, Winogrande
├── 代码生成 ── HumanEval, MBPP, SWE-bench
├── 推理能力 ── GSM8K, MATH, BBH
├── 安全性   ── ToxiGen, RealToxicityPrompts
└── 领域特定 ── MedQA, LegalBench, FinBench
```

### 1.3 关键术语

| 术语 | 定义 |
|------|------|
| **N-shot** | prompt 中提供的示例数量（0-shot、5-shot 等） |
| **Pass@k** | k 次尝试中至少有一次正确解的概率 |
| **数据污染** | 基准数据泄漏到训练数据中 |
| **基准饱和** | 模型接近满分，降低基准的区分度 |

---

## 2. 通用知识基准

### 2.1 MMLU（大规模多任务语言理解）

- **内容**：涵盖 STEM、人文、社会科学等 57 个学科
- **格式**：四选一多选题，14,042 道题
- **评估方式**：5-shot 准确率
- **重要性**：引用最广泛的通用知识基准

| 模型 | MMLU 分数 |
|------|----------|
| GPT-4 | 86.4% |
| Claude 3.5 Sonnet | 88.7% |
| Llama 3 70B | 82.0% |
| Mixtral 8x7B | 70.6% |

### 2.2 HellaSwag

- **内容**：需要常识推理的句子补全
- **格式**：四选一，10,042 道题
- **评估方式**：10-shot 准确率
- **关键洞察**：测试基础常识，而非单纯的事实记忆

### 2.3 ARC（AI2 推理挑战）

- **内容**：小学科学题
- **格式**：多选题；分为简单集（5,197 题）和挑战集（2,590 题）
- **评估方式**：25-shot 准确率（挑战集）
- **关键洞察**：挑战集过滤掉了可以通过简单检索回答的问题

### 2.4 TruthfulQA

- **内容**：测试模型是否生成真实答案，而非常见的错误认知
- **格式**：38 个类别共 817 个问题
- **评估方式**：人工评估的真实性 + 信息量
- **关键洞察**：更大的模型可能得分*更低*，因为它们更善于记住错误认知

### 2.5 Winogrande

- **内容**：需要常识推理的代词消解
- **格式**：二选一，44,000 道题
- **评估方式**：5-shot 准确率
- **示例**："奖杯放不进行李箱，因为*它*太[大/小]了。"

---

## 3. 代码基准

### 3.1 HumanEval

- **内容**：164 道手工编写的 Python 编程题
- **格式**：函数签名 + 文档字符串 → 完整实现
- **评估方式**：Pass@1, Pass@10, Pass@100（单元测试通过率）
- **来源**：OpenAI

```python
# HumanEval 示例题目
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """检查列表中是否有两个数字的差距小于给定阈值。
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
    True
    """
```

| 模型 | Pass@1 |
|------|--------|
| GPT-4 | 67.0% |
| Claude 3.5 Sonnet | 92.0% |
| Code Llama 34B | 48.8% |

### 3.2 MBPP（基础 Python 编程题）

- **内容**：974 道众包 Python 编程任务
- **格式**：任务描述 + 3 个测试用例 → 解答
- **评估方式**：Pass@1
- **与 HumanEval 的区别**：题目更简单，数据集更大

### 3.3 SWE-bench

- **内容**：来自 GitHub Issue 的真实软件工程任务
- **格式**：给定代码库 + Issue 描述 → 生成补丁
- **评估方式**：解决的 Issue 百分比（通过现有测试套件验证）
- **关键洞察**：测试真实世界的编码能力，而非孤立的函数生成

---

## 4. 推理基准

### 4.1 GSM8K（小学数学 8K）

- **内容**：8,500 道小学数学应用题
- **格式**：自然语言题目 → 逐步解答 + 数值答案
- **评估方式**：最终答案精确匹配
- **关键洞察**：测试多步算术推理；Chain-of-Thought 效果显著

| 模型 | GSM8K (CoT) |
|------|------------|
| GPT-4 | 92.0% |
| Llama 3 70B | 83.7% |
| Mistral 7B | 52.2% |

### 4.2 MATH

- **内容**：12,500 道竞赛级数学题（AMC, AIME 等）
- **格式**：题目 → LaTeX 解答
- **评估方式**：最终答案精确匹配
- **难度等级**：1-5（Level 5 = 竞赛难度）
- **关键洞察**：比 GSM8K 难得多；测试真正的数学推理能力

### 4.3 BBH（BIG-Bench Hard）

- **内容**：来自 BIG-Bench 的 23 个挑战任务，LLM 此前表现不佳
- **格式**：任务特定（逻辑推理、日期理解、因果判断等）
- **评估方式**：3-shot CoT 准确率
- **关键洞察**：专门选取"仅靠模型扩大规模无法解决"的任务

---

## 5. 安全基准

### 5.1 ToxiGen

- **内容**：针对 13 个少数群体的机器生成的有毒和良性语句
- **格式**：274,186 条语句 → 分类为有毒或良性
- **评估方式**：毒性分类准确率
- **关键洞察**：同时测试毒性检测和生成偏见

### 5.2 RealToxicityPrompts

- **内容**：来自网络文本的 100,000 个自然发生的 prompt
- **格式**：prompt → 衡量模型续写的毒性
- **评估方式**：期望最大毒性、毒性概率
- **关键洞察**：测试模型在看似无害的 prompt 下生成有毒内容的可能性

### 5.3 其他安全评测

| 基准 | 关注领域 |
|------|---------|
| **BBQ** | 问答中的社会偏见 |
| **WinoBias** | 共指消解中的性别偏见 |
| **CrowS-Pairs** | 刻板印象关联 |
| **HarmBench** | 对抗攻击鲁棒性 |

---

## 6. 排行榜

### 6.1 Open LLM Leaderboard（Hugging Face）

- **地址**：huggingface.co/spaces/open-llm-leaderboard
- **基准**：MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K
- **范围**：仅开源权重模型
- **更新**：社区持续提交
- **价值**：比较开源模型的首选参考

### 6.2 Chatbot Arena（LMSYS）

- **地址**：chat.lmsys.org
- **方法**：人类评审者进行匿名 A/B 对比
- **指标**：Elo 等级分（类似国际象棋）
- **范围**：开源和闭源模型
- **价值**：最能反映真实世界人类偏好

### 6.3 MTEB（大规模文本嵌入基准）

- **地址**：huggingface.co/spaces/mteb/leaderboard
- **聚焦**：嵌入模型评测
- **任务**：检索、分类、聚类、语义相似度、重排序
- **价值**：比较嵌入模型的标准（与 RAG 相关）

### 6.4 排行榜对比

| 排行榜 | 评估类型 | 覆盖模型 | 最适场景 |
|--------|---------|---------|---------|
| Open LLM Leaderboard | 自动化基准 | 开源权重 | 客观比较 |
| Chatbot Arena | 人类偏好 | 开源+闭源 | 真实世界质量 |
| MTEB | 自动化（嵌入） | 嵌入模型 | RAG/搜索选型 |
| AlpacaEval | LLM-as-judge | 开源+闭源 | 指令遵循能力 |

---

## 7. 实操指南 — lm-eval-harness

### 7.1 概述

[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 是评估语言模型的标准开源框架。支持 200+ 基准测试，被 Hugging Face Open LLM Leaderboard 采用。

### 7.2 安装

```bash
pip install lm-eval
```

### 7.3 基本用法

```bash
# 在 MMLU 上评估 HuggingFace 模型
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 8 \
    --output_path ./results/

# 在多个基准上评估
lm_eval --model hf \
    --model_args pretrained=mistralai/Mistral-7B-v0.1 \
    --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,winogrande,gsm8k \
    --batch_size auto \
    --output_path ./results/
```

### 7.4 评估 OpenAI 模型

```bash
# 使用 OpenAI API
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

# 打印结果
for task, metrics in results["results"].items():
    print(f"{task}: {metrics['acc,none']:.4f}")
```

### 7.6 自定义评测任务

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
# 运行自定义任务
lm_eval --model hf \
    --model_args pretrained=my-model \
    --tasks my_custom_qa \
    --include_path ./custom_tasks/
```

---

## 8. 最佳实践

### 8.1 基准数据污染

**问题**：如果基准数据出现在训练集中，分数会虚高且不可靠。

**检测方法**：
- 训练数据与基准之间的 N-gram 重叠分析
- 在测试集中插入金丝雀字符串（Canary string）
- 对留出集 vs 基准数据进行困惑度分析

**缓解措施**：
- 使用未公开的留出测试集
- 定期创建新的评估集
- 在报告分数时一并说明训练数据来源

### 8.2 报告标准

报告基准结果时，必须包含：

| 必填项 | 描述 |
|--------|------|
| 模型版本 | 精确的模型标识符和检查点 |
| N-shot 设置 | 使用的 few-shot 示例数量 |
| Prompt 格式 | 评估使用的确切模板 |
| 评估框架 | 工具和版本（如 lm-eval-harness v0.4.0） |
| 量化方式 | 如适用（如 4-bit, 8-bit） |
| 硬件 | GPU 类型和数量 |

### 8.3 自定义评测设计

当标准基准无法覆盖你的用例时：

1. **定义明确的成功标准**：对于你的应用，"好"是什么样的？
2. **构建领域特定测试集**：100-500 个带有标准答案的代表性示例
3. **使用多种评估方法**：
   - 自动化指标（精确匹配、F1、BLEU、ROUGE）
   - LLM-as-judge（GPT-4 / Claude 评分）
   - 人工评估（用于主观质量）
4. **版本管理测试集**：跟踪变更以防止意外污染
5. **报告置信区间**：使用 Bootstrap 采样确保统计显著性

### 8.4 常见陷阱

1. **选择性报告**：只报告模型表现优异的基准
2. **忽略方差**：不报告多次运行的标准差
3. **Prompt 敏感性**：不同 prompt 模板的结果可能相差 5-15%
4. **基准饱和**：模型达到 95%+ 分数后，基准的区分度大幅降低
5. **过度依赖排行榜**：MMLU 排名第一的模型可能在你的特定任务上表现不佳
6. **忽略延迟/成本**：基准分数不反映推理速度和成本效率
