# Pre-trained Language Models

**[English](README_EN.md) | [中文](README.md)**

## Overview

Pre-trained language models have revolutionized NLP by learning general language representations from large text corpora, then fine-tuning on specific tasks. This guide covers the major model families: BERT, GPT, and T5.

## BERT Family (Encoder-Only)

### BERT Architecture

**Bidirectional Encoder Representations from Transformers**

| Component | Specification |
|-----------|--------------|
| Architecture | Encoder-only Transformer |
| Objective | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) |
| Tokenization | WordPiece (30K vocab) |
| Position Encoding | Learned positional embeddings |

**Model Sizes**:
| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### Masked Language Modeling

**Objective**: Predict masked tokens from context

```
Input:  The [MASK] sat on the mat.
Target: cat
```

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Masked prediction
text = "The [MASK] sat on the mat."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Get top predictions for [MASK]
mask_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, mask_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(f"Predicted: {predicted_token}")  # "cat"
```

### BERT Variants

| Model | Key Improvement | Use Case |
|-------|----------------|----------|
| **RoBERTa** | Better training: more data, no NSP, dynamic masking | General NLP |
| **ALBERT** | Parameter sharing, factorized embeddings | Memory-constrained |
| **DistilBERT** | Knowledge distillation (40% smaller, 97% performance) | Efficiency |
| **ELECTRA** | Replaced token detection (more sample-efficient) | Pre-training |
| **DeBERTa** | Disentangled attention, enhanced mask decoder | SOTA encoder |

## GPT Family (Decoder-Only)

### GPT Architecture

**Generative Pre-trained Transformer**

| Component | Specification |
|-----------|--------------|
| Architecture | Decoder-only Transformer (causal) |
| Objective | Causal Language Modeling (CLM) |
| Tokenization | BPE/GPT-2 tokenizer |

**Evolution**:
| Model | Year | Params | Key Innovation |
|-------|------|--------|---------------|
| GPT | 2018 | 117M | First GPT |
| GPT-2 | 2019 | 1.5B | Zero-shot capability |
| GPT-3 | 2020 | 175B | In-context learning, few-shot |
| GPT-4 | 2023 | Unknown | Multimodal, reasoning |

### Causal Language Modeling

**Objective**: Predict next token given previous tokens

```
Context: The cat sat
Target: on
```

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Text generation
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors='pt')

# Generate
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

### Modern GPT Models

| Model | Developer | Params | Context | Notable Features |
|-------|-----------|--------|---------|-----------------|
| **GPT-3.5** | OpenAI | 175B | 4K | Chat-optimized |
| **GPT-4** | OpenAI | Unknown | 8K-32K | Multimodal, reasoning |
| **LLaMA** | Meta | 7B-65B | 2K-4K | Open weights, efficient |
| **LLaMA-2** | Meta | 7B-70B | 4K | Open commercial use |
| **CodeLLaMA** | Meta | 7B-34B | 4K-100K | Code specialization |
| **Mistral** | Mistral AI | 7B | 8K | Sliding window attention |
| **Mixtral** | Mistral AI | 8×7B | 32K | Sparse MoE architecture |

## T5 Family (Encoder-Decoder)

### T5 Architecture

**Text-to-Text Transfer Transformer**

Key principle: Frame all NLP tasks as text-to-text problems

```
Input:  translate English to German: The house is wonderful.
Output: Das Haus ist wunderbar.

Input:  cola sentence: The movie was boring.
Output: unacceptable
```

| Model | Params | Architecture |
|-------|--------|--------------|
| T5-Small | 60M | 6 layers each |
| T5-Base | 220M | 12 layers each |
| T5-Large | 770M | 24 layers each |
| T5-3B | 3B | 24 layers each |
| T5-11B | 11B | 24 layers each |

### Denoising Objective

**Span Corruption**: Replace consecutive spans with unique sentinel tokens

```
Original: Thank you for inviting me to your party last week.
Corrupted: Thank you <X> me to your party <Y>.
Target: <X> for inviting <Y> last week.
```

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Translation task
input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # "Das Haus ist wunderbar."
```

## Transfer Learning Strategies

### 1. Feature Extraction

**Freeze pre-trained weights, only train classification head**

```python
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained BERT (frozen)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Trainable classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask)
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
```

**Best for**:
- Small datasets (< 1K examples)
- Quick prototyping
- Limited compute

### 2. Fine-Tuning

**Update all parameters with small learning rate**

```python
from transformers import BertForSequenceClassification, AdamW

# Load with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Optimizer with different learning rates
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},  # Lower for BERT
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # Higher for head
])

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**Best for**:
- Medium to large datasets
- When task differs significantly from pre-training
- Maximum performance

### 3. Layer-wise Learning Rate Decay

```python
# Gradually decrease LR for earlier layers
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.bert.embeddings.named_parameters() 
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': 1e-5  # Lowest for embeddings
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
        'lr': 3e-5  # Highest for top layers
    },
    {
        'params': [p for n, p in model.classifier.named_parameters()],
        'weight_decay': 0.01,
        'lr': 1e-3  # Highest for classifier
    }
]

optimizer = AdamW(optimizer_grouped_parameters)
```

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| **Classification** | RoBERTa, DeBERTa | Strong encoder representations |
| **Named Entity Recognition** | BERT, RoBERTa | Token-level classification |
| **Question Answering** | RoBERTa, ELECTRA | Good span extraction |
| **Text Generation** | GPT-4, LLaMA-2, Mistral | Autoregressive generation |
| **Chat/Dialogue** | GPT-3.5, LLaMA-2-Chat | Instruction-tuned |
| **Code Generation** | CodeLLaMA, GPT-4 | Code pre-training |
| **Summarization** | T5, BART | Encoder-decoder architecture |
| **Translation** | T5, mT5 | Text-to-text framework |
| **Multilingual** | XLM-R, mBERT | Cross-lingual training |
| **Long Documents** | Longformer, BigBird | Efficient long attention |

## Fine-Tuning Best Practices

### 1. Learning Rate

| Model Size | Typical LR Range |
|------------|-----------------|
| Base (110M) | 2e-5 to 5e-5 |
| Large (340M) | 1e-5 to 3e-5 |
| 1B+ | 1e-5 to 2e-5 |

### 2. Batch Size

- Larger batches (32-128) generally better for fine-tuning
- Use gradient accumulation if GPU memory limited

### 3. Epochs

| Dataset Size | Recommended Epochs |
|--------------|-------------------|
| < 1K | 10-20 |
| 1K - 10K | 3-5 |
| > 10K | 2-3 |

### 4. Early Stopping

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
            print(f"Early stopping at epoch {epoch}")
            break
```

## Advanced Techniques

### 1. Prompt Tuning

**Soft prompts**: Trainable continuous vectors prepended to input

```python
class PromptTunedModel(nn.Module):
    def __init__(self, model_name, num_tokens=20):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, self.model.config.hidden_size)
        )
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        # Expand prompt for batch
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)
        
        # Get input embeddings
        inputs_embeds = self.model.embeddings(input_ids)
        
        # Concatenate prompt + input
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        
        # Adjust attention mask
        prompt_mask = torch.ones(batch_size, prompts.size(1))
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs
```

### 2. Adapter Layers

**Small trainable modules inserted between frozen layers**

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
        return x + residual  # Residual connection

# Insert adapters into BERT
for layer in model.bert.encoder.layer:
    layer.output.adapters = Adapter(768)
```

### 3. Progressive Layer Freezing

**Gradually unfreeze layers during training**

```python
def progressive_unfreeze(model, epoch, total_epochs):
    """Unfreeze from top to bottom"""
    num_layers = len(model.bert.encoder.layer)
    layers_to_unfreeze = int(num_layers * (epoch / total_epochs))
    
    # Freeze all first
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Unfreeze top layers
    for i in range(num_layers - layers_to_unfreeze, num_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = True
```

## Evaluation Metrics

### Classification
- **Accuracy**: Overall correctness
- **F1-Score**: Balance of precision and recall
- **AUC-ROC**: Ranking quality

### Generation
- **BLEU**: N-gram overlap with reference
- **ROUGE**: Recall-oriented overlap
- **Perplexity**: Model confidence

### Similarity
- **Cosine Similarity**: Vector space similarity
- **BERTScore**: Contextual embedding similarity

## Common Pitfalls

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Catastrophic Forgetting** | Poor generalization | Use lower LR, more regularization |
| **Overfitting** | High train acc, low val acc | Add dropout, early stopping, more data |
| **Underfitting** | Both accuracies low | Train longer, higher LR, unfreeze more |
| **Long Input** | Truncation hurts | Use Longformer, sliding window, or chunking |
| **Imbalanced Data** | Biased predictions | Class weights, oversampling, focal loss |

---

**Previous**: [Transformer Architecture](../transformer-architecture/README.md) | **Next**: [Pre-training](../../04-LLM-Core/pre-training/README.md)