# 为什么"先读万卷书"比"直接考试"更有效？—— 预训练语言模型
[English](README_EN.md) | [中文](README.md)

## 这个问题从哪来

> 2018 年之前，NLP 任务的主流模式是「针对每个任务从头训练一个模型」。这意味着你有 100 个任务，就要训练 100 个模型，而且每个模型都只能看到对应任务那点可怜的数据。
> 2018 年，三件事同时发生：ELMo 证明上下文词向量比静态词向量强大得多；GPT-1 证明生成式预训练可以迁移到下游任务；BERT 证明双向编码器+掩码语言建模在理解任务上几乎通吃。这一年，"预训练 + 微调"范式正式确立，NLP 从「作坊式建模」进入了「工业化生产」时代。

## 学习目标

完成本章后，你应能回答：

1. BERT、GPT、T5 三家在预训练目标和架构选择上有什么区别？
2. 特征提取（冻结 backbone）和微调（更新全部参数）各适合什么场景？
3. 面对具体任务，如何选择合适的预训练模型和微调策略？

---

## 1. 直觉

想象你要准备法律职业资格考试。

**方案 A**：直接买一套法考真题，闭门刷题。优点是针对性强；缺点是你没有法学基础，很多概念根本没见过，刷再多也记不住。

**方案 B**：先花两年时间系统读完法学本科教材（预训练），建立完整的知识体系，然后再花两周刷真题并重点补习考试技巧（微调）。结果通常更好，而且你学到的法律知识还能用来写合同、做咨询、参加辩论——迁移能力很强。

预训练语言模型的核心直觉就在于此：先让模型在海量无标注文本上学习通用语言规律（语法、语义、常识、推理模式），然后在具体任务上用少量标注数据做针对性调整。这个「通用 → 专用」的分层策略，让小样本任务也能达到很好的效果。

> 你要记住：预训练解决的是"通用语言能力"，微调解决的是"任务对齐"。两者缺一不可。

---

## 2. 机制

### 2.1 BERT：双向理解专家

BERT（Bidirectional Encoder Representations from Transformers）只使用 Transformer 的编码器部分，训练目标是**掩码语言建模（MLM）**：随机遮住输入中 15% 的 token，让模型根据双向上下文预测被遮的内容。

```
输入:  The [MASK] sat on the mat.
目标:  cat
```

由于编码器是双向的，BERT 擅长理解类任务：文本分类、命名实体识别、问答抽取、语义相似度。它不擅长生成，因为没有自回归解码机制。

**BERT 变体速览**：

| 模型 | 关键改进 | 适用场景 |
|------|---------|---------|
| RoBERTa | 更大语料、去掉 NSP、动态掩码 | 通用 NLP 理解任务 |
| ALBERT | 参数共享、分解嵌入 | 资源受限场景 |
| DistilBERT | 知识蒸馏（体积 -40%，性能 -3%）| 边缘部署、低延迟 |
| ELECTRA | 替换 token 检测（样本效率更高）| 预训练阶段 |
| DeBERTa | 解耦注意力、增强掩码解码器 | 需要最强编码器时 |

### 2.2 GPT：单向生成专家

GPT（Generative Pre-trained Transformer）只使用 Transformer 的解码器部分，训练目标是**因果语言建模（CLM / Next Token Prediction）**：给定前面的 token，预测下一个 token。

```
上下文: The cat sat
目标:    on
```

这种自回归特性让 GPT 家族天然适合文本生成、对话、代码补全、推理链（Chain-of-Thought）。从 GPT-1（117M）到 GPT-4（多模态），核心架构没变，变的只是规模、数据和对齐策略。

**现代代表模型**：GPT-4、LLaMA-2、Mistral、CodeLLaMA、Mixtral（MoE）

### 2.3 T5：统一文本到文本框架

T5（Text-to-Text Transfer Transformer）使用完整的 Encoder-Decoder 架构，核心创新是把**所有 NLP 任务都统一成文本到文本问题**：分类不是输出标签，而是输出文字；翻译、摘要、问答都是输入一段文本、输出一段文本。

训练目标是**跨度破坏（Span Corruption）**：用唯一的哨兵 token 替换输入中的连续片段，解码器负责生成被替换的内容。

```
原始: Thank you for inviting me to your party last week.
损坏: Thank you <X> me to your party <Y>.
目标: <X> for inviting <Y> last week.
```

T5 及其变体（mT5、Flan-T5）是条件生成任务（翻译、摘要、改写）的首选。

### 2.4 三家族对比

| 维度 | BERT | GPT | T5 |
|------|------|-----|-----|
| 架构 | Encoder-only | Decoder-only | Encoder-Decoder |
| 预训练目标 | MLM（掩码预测）| CLM（下一个 token）| Span Corruption |
| 擅长任务 | 理解、分类、抽取 | 生成、对话、推理 | 翻译、摘要、条件生成 |
| 代表 | BERT/RoBERTa/DeBERTa | GPT-4/LLaMA/Mistral | T5/BART/mT5 |

> 你要记住：选模型先看任务是"理解"还是"生成"——理解用 BERT 系，生成用 GPT 系，条件生成用 T5 系。

### 2.5 渐进式实现

**Step 1 · 解决最小可运行理解任务：BERT 掩码预测**

```python
# 使用预训练 BERT 预测 [MASK] 位置最可能的 token
# 验证双向上下文对理解任务的价值
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The [MASK] sat on the mat."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

mask_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, mask_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(f"预测: {predicted_token}")  # cat
```

**Step 2 · 解决输入约束与生成边界：GPT 文本生成**

```python
# 使用 GPT-2 进行自回归文本生成
# 验证因果语言建模的生成能力
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Step 3 · 解决任务范式对比：T5 翻译任务**

```python
# 使用 T5 进行英德翻译
# 验证文本到文本统一框架
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Step 4 · 解决生产级选择：特征提取 vs 微调**

```python
# 方案 A：冻结 BERT，只训练分类头（特征提取）
from transformers import BertModel
import torch.nn as nn

class BERTFeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0])

# 方案 B：微调全部参数，但使用不同学习率
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

---

## 3. 工程陷阱

1. **灾难性遗忘**
   现象：预训练模型在下游任务上微调时，学习率过大导致模型丢失通用语言能力。
   处置：微调学习率通常比预训练低 10-100 倍（如 2e-5）；大数据集可以解冻更多层，小数据集优先特征提取或 PEFT。

2. **小数据集盲目选择全参数微调**
   现象：只有几百条样本却更新上亿参数，严重过拟合。
   处置：样本 < 1K 时优先特征提取；1K-10K 可尝试轻量微调 + 早停；大数据集再全参数微调。

3. **输入过长导致截断，性能受损**
   现象：BERT 默认 max_length=512，长文档被截断后丢失后半部分关键信息。
   处置：理解任务可用 Longformer/BigBird；生成任务可尝试滑动窗口或分块策略。

4. **数据不平衡未处理**
   现象：分类任务中多数类占 95%，模型倾向于全预测多数类，准确率虚高。
   处置：使用 Focal Loss、类别权重、过采样，或以 F1 代替 Accuracy 作为主要指标。

---

## 演进笔记

> **预训练范式的演进**：从静态词向量（Word2Vec）→ 上下文词向量（ELMo）→ 预训练 + 微调（BERT/GPT-1）→ 更大规模 + 提示工程（GPT-3）→ 参数高效微调（Prompt Tuning、Adapter、LoRA）→ 对齐与人类反馈（RLHF）。
>
> 这一演进的核心线索是：随着模型变大，"更新全部参数"的微调成本越来越高，于是研究重心从"怎么训练大模型"转向"怎么用大模型做更多事，同时少花钱"。
>
> **留下的新问题**：当预训练模型大到 175B 参数时，全参数微调已经不可行。如何用 1% 甚至 0.1% 的参数完成下游任务适配？这催生了参数高效微调（PEFT）和对齐技术。

→ 下一阶段：[汇流：规模与多模态](../../scale-multimodal/README.md)

---

**上一章**：[Transformer 架构](../transformer-architecture/README.md) | **下一章**：[汇流：规模与多模态](../../scale-multimodal/README.md)
