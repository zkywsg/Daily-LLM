[English](README_EN.md) | [中文](README.md)

# Why Is "Reading Widely First" More Effective Than "Taking the Exam Cold"? — Pre-trained Language Models

## Where This Problem Came From

> Before 2018, the dominant paradigm in NLP was "train a separate model from scratch for every task." This meant 100 tasks required 100 models, each seeing only the small amount of data available for that specific task.
> In 2018, three breakthroughs happened simultaneously: ELMo showed that contextual word vectors dramatically outperformed static embeddings; GPT-1 proved that generative pre-training could transfer to downstream tasks; and BERT demonstrated that a bidirectional encoder with masked language modeling could dominate almost all understanding tasks. That year, the "pre-train + fine-tune" paradigm was established, and NLP moved from "artisanal modeling" to "industrial production."

## Learning Goals

After completing this chapter, you should be able to answer:

1. How do BERT, GPT, and T5 differ in pre-training objectives and architecture choices?
2. When should you use feature extraction (frozen backbone) versus fine-tuning (updating all parameters)?
3. Given a specific task, how do you choose the right pre-trained model and fine-tuning strategy?

---

## 1. Intuition

Imagine you are preparing for a bar exam.

**Plan A**: Buy a set of past exam papers and study them in isolation. The upside is task-specific focus; the downside is that without foundational legal knowledge, many concepts will be completely foreign no matter how many papers you drill.

**Plan B**: Spend two years systematically reading law school textbooks (pre-training) to build a comprehensive knowledge base, then spend two weeks doing practice papers and exam technique drills (fine-tuning). The result is usually much better, and the legal knowledge you acquired can also be used for writing contracts, giving consultations, and participating in debates — strong transferability.

The core intuition of pre-trained language models is exactly this: first let the model learn general language patterns (grammar, semantics, common sense, reasoning) from massive unlabeled text, then make task-specific adjustments with a small amount of labeled data. This "general → specialized" layered strategy allows even low-resource tasks to achieve strong results.

> Key takeaway: pre-training solves "general language ability," while fine-tuning solves "task alignment." Neither alone is enough.

---

## 2. Mechanism

### 2.1 BERT: The Bidirectional Understanding Expert

BERT (Bidirectional Encoder Representations from Transformers) uses only the Transformer encoder. Its training objective is **Masked Language Modeling (MLM)**: randomly mask 15% of the input tokens and let the model predict the masked content using bidirectional context.

```
Input:  The [MASK] sat on the mat.
Target: cat
```

Because the encoder is bidirectional, BERT excels at understanding tasks: text classification, named entity recognition, extractive question answering, and semantic similarity. It is not well-suited for generation because it lacks an autoregressive decoding mechanism.

**BERT variants at a glance**:

| Model | Key Improvement | Best For |
|-------|-----------------|----------|
| RoBERTa | Larger corpus, no NSP, dynamic masking | General NLP understanding |
| ALBERT | Parameter sharing, factorized embeddings | Resource-constrained scenarios |
| DistilBERT | Knowledge distillation (-40% size, -3% performance) | Edge deployment, low latency |
| ELECTRA | Replaced token detection (higher sample efficiency) | Pre-training stage |
| DeBERTa | Disentangled attention, enhanced mask decoder | When you need the strongest encoder |

### 2.2 GPT: The Unidirectional Generation Expert

GPT (Generative Pre-trained Transformer) uses only the Transformer decoder. Its training objective is **Causal Language Modeling (CLM / Next Token Prediction)**: given previous tokens, predict the next one.

```
Context: The cat sat
Target:  on
```

This autoregressive nature makes the GPT family naturally suited for text generation, dialogue, code completion, and chain-of-thought reasoning. From GPT-1 (117M) to GPT-4 (multimodal), the core architecture has not changed — only scale, data, and alignment strategies have.

**Modern representative models**: GPT-4, LLaMA-2, Mistral, CodeLLaMA, Mixtral (MoE)

### 2.3 T5: The Unified Text-to-Text Framework

T5 (Text-to-Text Transfer Transformer) uses the full Encoder-Decoder architecture. Its core innovation is unifying **all NLP tasks as text-to-text problems**: classification does not output a label but a text string; translation, summarization, and question answering all take one text segment as input and produce another as output.

The training objective is **Span Corruption**: replace contiguous spans in the input with unique sentinel tokens, and let the decoder generate the replaced content.

```
Original: Thank you for inviting me to your party last week.
Corrupt:  Thank you <X> me to your party <Y>.
Target:   <X> for inviting <Y> last week.
```

T5 and its variants (mT5, Flan-T5) are the go-to choice for conditional generation tasks (translation, summarization, rewriting).

### 2.4 Three Families Compared

| Dimension | BERT | GPT | T5 |
|-----------|------|-----|-----|
| Architecture | Encoder-only | Decoder-only | Encoder-Decoder |
| Pre-training objective | MLM (masked prediction) | CLM (next token) | Span Corruption |
| Best at | Understanding, classification, extraction | Generation, dialogue, reasoning | Translation, summarization, conditional generation |
| Representatives | BERT/RoBERTa/DeBERTa | GPT-4/LLaMA/Mistral | T5/BART/mT5 |

> Key takeaway: choose the model family based on whether the task is "understanding" or "generation" — BERT for understanding, GPT for generation, T5 for conditional generation.

### 2.5 Progressive Implementation

**Step 1 · Solve the smallest runnable understanding task: BERT masked prediction**

```python
# Use pre-trained BERT to predict the most likely token at [MASK]
# Validate the value of bidirectional context for understanding tasks
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
print(f"Prediction: {predicted_token}")  # cat
```

**Step 2 · Solve input constraints and generation boundaries: GPT text generation**

```python
# Use GPT-2 for autoregressive text generation
# Validate the generation capability of causal language modeling
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

**Step 3 · Solve task-paradigm comparison: T5 translation**

```python
# Use T5 for English-to-German translation
# Validate the unified text-to-text framework
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Step 4 · Solve production choices: feature extraction vs fine-tuning**

```python
# Plan A: Freeze BERT and only train the classification head (feature extraction)
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

# Plan B: Fine-tune all parameters with different learning rates
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

---

## 3. Engineering Pitfalls

1. **Catastrophic forgetting**
   Symptom: during downstream fine-tuning, the model loses its general language ability because the learning rate is too high.
   Fix: fine-tuning learning rates are typically 10-100x lower than pre-training (e.g. 2e-5); use feature extraction or PEFT for large datasets with limited resources.

2. **Blindly choosing full-parameter fine-tuning on small datasets**
   Symptom: updating hundreds of millions of parameters with only a few hundred samples leads to severe overfitting.
   Fix: samples < 1K → prefer feature extraction; 1K-10K → light fine-tuning + early stopping; large datasets → full-parameter fine-tuning.

3. **Input truncation hurting performance**
   Symptom: BERT's default max_length=512 truncates long documents, discarding critical information in the latter half.
   Fix: for understanding tasks use Longformer/BigBird; for generation use sliding windows or chunking strategies.

4. **Ignoring class imbalance**
   Symptom: one class dominates 95% of the data, so the model predicts the majority class every time and accuracy looks deceptively high.
   Fix: use Focal Loss, class weights, oversampling, or replace Accuracy with F1 as the primary metric.

---

## Evolution Notes

> **The evolution of the pre-training paradigm**: static word vectors (Word2Vec) → contextual word vectors (ELMo) → pre-train + fine-tune (BERT/GPT-1) → larger scale + prompt engineering (GPT-3) → parameter-efficient fine-tuning (Prompt Tuning, Adapters, LoRA) → alignment with human feedback (RLHF).
>
> The central thread of this evolution is: as models grow larger, the cost of "updating all parameters" becomes prohibitive, so research focus shifts from "how to train big models" to "how to do more with big models while spending less."
>
> **New question left behind**: when pre-trained models reach 175B parameters, full fine-tuning is no longer feasible. How can we adapt them to downstream tasks using 1% or even 0.1% of the parameters? This gave rise to Parameter-Efficient Fine-Tuning (PEFT) and alignment techniques.

→ Next Phase: [Scale & Multimodal](../../scale-multimodal/README_EN.md)

---

**Previous**: [Transformer Architecture](../transformer-architecture/README_EN.md) | **Next**: [Scale & Multimodal](../../scale-multimodal/README_EN.md)
