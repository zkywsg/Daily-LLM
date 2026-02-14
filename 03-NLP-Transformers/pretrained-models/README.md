[English](README_EN.md) | [中文](README.md)

# 预训练语言模型

## 概述

预训练语言模型通过从大规模文本语料库中学习通用语言表示，然后在特定任务上进行微调，彻底改变了自然语言处理（NLP）。本指南涵盖主要的模型家族：BERT、GPT 和 T5。

## BERT 家族 (仅编码器)

### BERT 架构

**Transformer 的双向编码器表示**

| 组件 | 规范 |
|-----------|--------------|
| 架构 | 仅编码器 Transformer |
| 目标 | 掩码语言建模 (MLM) + 下一句预测 (NSP) |
| 分词 | WordPiece (30K 词汇表) |
| 位置编码 | 学习到的位置嵌入 |

**模型规格**：
| 变体 | 层数 | 隐藏层 | 头数 | 参数量 |
|---------|--------|--------|-------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### 掩码语言建模

**目标**：从上下文预测被掩码的 token

```
输入:  The <mask> sat on the mat.
目标: cat
```

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 掩码预测
text = "The <mask> sat on the mat."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 获取 <mask> 的前几个预测
mask_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, mask_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(f"预测: {predicted_token}")  # "cat"
```

### BERT 变体

| 模型 | 关键改进 | 应用场景 |
|-------|----------------|----------|
| **RoBERTa** | 更好的训练：更多数据、无 NSP、动态掩码 | 通用 NLP |
| **ALBERT** | 参数共享、分解嵌入 | 内存受限场景 |
| **DistilBERT** | 知识蒸馏（40% 更小，97% 性能） | 效率优先 |
| **ELECTRA** | 替换 token 检测（样本效率更高） | 预训练 |
| **DeBERTa** | 解耦注意力、增强掩码解码器 | 最先进编码器 |

## GPT 家族 (仅解码器)

### GPT 架构

**生成式预训练 Transformer**

| 组件 | 规范 |
|-----------|--------------|
| 架构 | 仅解码器 Transformer（因果） |
| 目标 | 因果语言建模 (CLM) |
| 分词 | BPE/GPT-2 分词器 |

**演进历程**：
| 模型 | 年份 | 参数量 | 关键创新 |
|-------|------|--------|---------------|
| GPT | 2018 | 117M | 第一个 GPT |
| GPT-2 | 2019 | 1.5B | 零样本能力 |
| GPT-3 | 2020 | 175B | 上下文学习、少样本 |
| GPT-4 | 2023 | 未知 | 多模态、推理 |

### 因果语言建模

**目标**：给定之前的 token 预测下一个 token

```
上下文: The cat sat
目标: on
```

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 文本生成
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors='pt')

# 生成
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### 现代 GPT 模型

| 模型 | 开发者 | 参数量 | 上下文 | 显著特征 |
|-------|-----------|--------|---------|-----------------|
| **GPT-3.5** | OpenAI | 175B | 4K | 聊天优化 |
| **GPT-4** | OpenAI | 未知 | 8K-32K | 多模态、推理 |
| **LLaMA** | Meta | 7B-65B | 2K-4K | 开放权重、高效 |
| **LLaMA-2** | Meta | 7B-70B | 4K | 开放商业使用 |
| **CodeLLaMA** | Meta | 7B-34B | 4K-100K | 代码专精 |
| **Mistral** | Mistral AI | 7B | 8K | 滑动窗口注意力 |
| **Mixtral** | Mistral AI | 8×7B | 32K | 稀疏 MoE 架构 |

## T5 家族 (编码器-解码器)

### T5 架构

**文本到文本迁移 Transformer**

核心原则：将所有 NLP 任务构建为文本到文本问题

```
输入:  translate English to German: The house is wonderful.
输出: Das Haus ist wunderbar.

输入:  cola sentence: The movie was boring.
输出: unacceptable
```

| 模型 | 参数量 | 架构 |
|-------|--------|--------------|
| T5-Small | 60M | 各 6 层 |
| T5-Base | 220M | 各 12 层 |
| T5-Large | 770M | 各 24 层 |
| T5-3B | 3B | 各 24 层 |
| T5-11B | 11B | 各 24 层 |

### 去噪目标

**跨度破坏**：用唯一的哨兵 token 替换连续的跨度

```
原始: Thank you for inviting me to your party last week.
损坏: Thank you <X> me to your party <Y>.
目标: <X> for inviting <Y> last week.
```

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载模型
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# 翻译任务
input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

# 生成
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # "Das Haus ist wunderbar."
```

## 迁移学习策略

### 1. 特征提取

**冻结预训练权重，仅训练分类头**

```python
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载预训练 BERT（冻结）
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False

        # 可训练的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask)

        # 使用 [CLS] token 表示
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
```

**适用于**：
- 小数据集（< 1K 样本）
- 快速原型
- 计算资源有限

### 2. 微调

**使用小学习率更新所有参数**

```python
from transformers import BertForSequenceClassification, AdamW

# 加载带分类头的模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# 使用不同学习率的优化器
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},  # BERT 使用较低学习率
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # 分类头使用较高学习率
])

# 训练循环
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**适用于**：
- 中到大数据集
- 任务与预训练有显著差异
- 追求最佳性能

### 3. 层级学习率衰减

```python
 # 逐步降低早期层的学习率
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.bert.embeddings.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': 1e-5  # 嵌入层使用最低学习率
    },
    {
        'params': [p for n, p in model.bert.encoder.layer[:6].named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': 2e-5
    },
    {
        'params': [p for n, p in model.bert.encoder.layer[6:].named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': 3e-5  # 顶层使用最高学习率
    },
    {
        'params': [p for n, p in model.classifier.named_parameters()],
        'weight_decay': 0.01,
        'lr': 1e-3  # 分类器使用最高学习率
    }
]

optimizer = AdamW(optimizer_grouped_parameters)
```

## 模型选择指南

| 使用场景 | 推荐模型 | 原因 |
|----------|------------------|--------|
| **分类** | RoBERTa, DeBERTa | 强大的编码器表示 |
| **命名实体识别** | BERT, RoBERTa | Token 级分类 |
| **问答** | RoBERTa, ELECTRA | 良好的跨度提取 |
| **文本生成** | GPT-4, LLaMA-2, Mistral | 自回归生成 |
| **聊天/对话** | GPT-3.5, LLaMA-2-Chat | 指令微调 |
| **代码生成** | CodeLLaMA, GPT-4 | 代码预训练 |
| **摘要** | T5, BART | 编码器-解码器架构 |
| **翻译** | T5, mT5 | 文本到文本框架 |
| **多语言** | XLM-R, mBERT | 跨语言训练 |
| **长文档** | Longformer, BigBird | 高效的长注意力 |

## 微调最佳实践

### 1. 学习率

| 模型规模 | 典型学习率范围 |
|------------|-----------------|
| Base (110M) | 2e-5 到 5e-5 |
| Large (340M) | 1e-5 到 3e-5 |
| 1B+ | 1e-5 到 2e-5 |

### 2. 批次大小

- 较大的批次（32-128）通常对微调更好
- 如果 GPU 内存有限，使用梯度累积

### 3. 训练轮数

| 数据集大小 | 推荐轮数 |
|--------------|-------------------|
| < 1K | 10-20 |
| 1K - 10K | 3-5 |
| > 10K | 2-3 |

### 4. 早停

```python
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"在第 {epoch} 轮早停")
            break
```

## 高级技术

### 1. 提示微调

**软提示**：在输入前添加可训练的连续向量

```python
class PromptTunedModel(nn.Module):
    def __init__(self, model_name, num_tokens=20):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, self.model.config.hidden_size)
        )
        # 冻结基础模型
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        # 为批次扩展提示
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)

        # 获取输入嵌入
        inputs_embeds = self.model.embeddings(input_ids)

        # 拼接提示 + 输入
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        # 调整注意力掩码
        prompt_mask = torch.ones(batch_size, prompts.size(1))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs
```

### 2. 适配器层

**在冻结层之间插入小型可训练模块**

```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # 残差连接

# 将适配器插入 BERT
for layer in model.bert.encoder.layer:
    layer.output.adapters = Adapter(768)
```

### 3. 渐进式层解冻

**训练期间逐步解冻层**

```python
def progressive_unfreeze(model, epoch, total_epochs):
    """从上到下解冻"""
    num_layers = len(model.bert.encoder.layer)
    layers_to_unfreeze = int(num_layers * (epoch / total_epochs))

    # 首先冻结所有层
    for param in model.bert.parameters():
        param.requires_grad = False

    # 解冻顶层
    for i in range(num_layers - layers_to_unfreeze, num_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = True
```

## 评估指标

### 分类
- **准确率 (Accuracy)**：整体正确性
- **F1 分数**：精确率和召回率的平衡
- **AUC-ROC**：排序质量

### 生成
- **BLEU**：与参考的 N-gram 重叠度
- **ROUGE**：召回率导向的重叠度
- **困惑度**：模型置信度

### 相似度
- **余弦相似度**：向量空间相似性
- **BERTScore**：上下文嵌入相似性

## 常见陷阱

| 问题 | 症状 | 解决方案 |
|-------|---------|----------|
| **灾难性遗忘** | 泛化能力差 | 使用更低学习率、更多正则化 |
| **过拟合** | 训练准确率高，验证准确率低 | 添加 dropout、早停、更多数据 |
| **欠拟合** | 两者准确率都低 | 训练更长时间、更高学习率、解冻更多层 |
| **输入过长** | 截断损害性能 | 使用 Longformer、滑动窗口或分块 |
| **数据不平衡** | 预测偏差 | 类权重、过采样、Focal Loss |

---

**上一章**: [Transformer 架构](../transformer-architecture/README.md) | **下一章**: [预训练](../../04-LLM-Core/pre-training/README.md)
