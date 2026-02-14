# Multimodal Models

**[English](README_EN.md) | [中文](README.md)**

## Overview

Multimodal models process and generate content across multiple modalities (text, images, audio, video). This guide focuses on vision-language models that combine computer vision with language understanding.

## Vision-Language Architecture Patterns

### 1. Dual-Encoder (Contrastive)

**Separate encoders for each modality with alignment in shared space:**

```
Image → Image Encoder → Image Embedding
                              ↓ (contrastive loss)
Text  → Text Encoder  → Text Embedding
```

**Examples**: CLIP, ALIGN

### 2. Fusion Encoder

**Early/intermediate fusion of modalities:**

```
Image → Vision Encoder ─┐
                        ├→ Fusion Layers → Decoder → Output
Text  → Text Encoder  ─┘
```

**Examples**: VisualBERT, ViLBERT, UNITER

### 3. Encoder-Decoder with Vision

**Vision encoder + text decoder (autoregressive):**

```
Image → Vision Encoder → Projector → Text Decoder → Caption/Response
```

**Examples**: BLIP, BLIP-2, LLaVA, GPT-4V

### 4. Flamingo-Style (Perceiver Resampler)

**Gated cross-attention in frozen LLM:**

```
Image Patches → Perceiver Resampler → Visual Tokens
                                           ↓
Text Tokens → Frozen LLM with Gated Cross-Attention → Output
```

**Examples**: Flamingo, OpenFlamingo, IDEFICS

## CLIP (Contrastive Language-Image Pre-training)

### Architecture

| Component | Specification |
|-----------|--------------|
| Image Encoder | ViT or ResNet |
| Text Encoder | Transformer |
| Projection | Linear layers to shared space |
| Training | Contrastive learning on (image, text) pairs |

### Contrastive Learning

**Objective**: Maximize similarity for matched pairs, minimize for unmatched

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        """
        image_features: (batch_size, embed_dim)
        text_features: (batch_size, embed_dim)
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2

# Using HuggingFace CLIP
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image-text similarity
image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a car"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Compute similarity
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)  # Highest for "a photo of a cat"
```

### Zero-Shot Classification

```python
def zero_shot_classify(image, class_names, model, processor):
    """Classify image without training"""
    # Create prompts
    texts = [f"a photo of a {name}" for name in class_names]
    
    # Process
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    
    # Get predictions
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=0)
    
    # Return top predictions
    top_probs, top_indices = probs.topk(5)
    return [(class_names[i], p.item()) for i, p in zip(top_indices, top_probs)]

# Usage
results = zero_shot_classify(image, ["cat", "dog", "bird", "fish"], model, processor)
```

### Image Retrieval

```python
def build_image_index(images, model, processor):
    """Build searchable image index"""
    image_features = []
    
    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
        image_features.append(features)
    
    return torch.cat(image_features, dim=0)

def search_images(query_text, image_index, model, processor, top_k=5):
    """Search images by text query"""
    # Encode query
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)
    
    # Compute similarities
    similarities = torch.matmul(text_features, image_index.T)
    
    # Get top matches
    top_scores, top_indices = similarities[0].topk(top_k)
    return top_indices.tolist(), top_scores.tolist()
```

## BLIP (Bootstrapping Language-Image Pre-training)

### Unified Architecture

BLIP unifies vision-language understanding and generation:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image captioning
image = Image.open("beach.jpg")
inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs, max_length=50)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)  # "a group of people walking on a beach with surfboards"
```

### Bootstrapped Training

BLIP uses a caption filter to bootstrap training data:

```
Raw Web Data → Noisy (Image, Text) Pairs
                    ↓
            ┌──────┴──────┐
            ↓             ↓
       Text Encoder    Image-grounded Text Decoder
            ↓             ↓
     ITC Loss         LM Loss
            ↓             ↓
       Image-Text     Caption
       Contrastive    Generation
            ↓             ↓
            └──────┬──────┘
                   ↓
            Caption Filter (Remove noisy captions)
                   ↓
            Bootstrapped Training Data
```

## BLIP-2: Efficient Vision-Language Pre-training

### Q-Former (Querying Transformer)

**Bridge frozen image encoder and frozen LLM:**

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load BLIP-2 (OPT-2.7B)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
)

# Visual question answering
image = Image.open("kitchen.jpg")
prompt = "Question: What appliance is in the image? Answer:"

inputs = processor(image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)  # "refrigerator"
```

### Two-Stage Training

**Stage 1: Vision-Language Representation Learning**
- Train Q-Former with frozen image encoder
- Use ITC, ITM, and image-grounded text generation objectives

**Stage 2: Vision-to-Language Generative Learning**
- Connect Q-Former to frozen LLM (OPT or Flan-T5)
- Train to generate text conditioned on visual queries

## LLaVA (Large Language and Vision Assistant)

### Architecture

**Vision Encoder + Projection Layer + LLM:**

```python
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor

class LLaVAModel:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = LlavaProcessor.from_pretrained(model_name)
    
    def generate_response(self, image, prompt, max_length=512):
        """Generate text response to image + prompt"""
        # Format prompt with image token
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract assistant response
        return response.split("ASSISTANT:")[-1].strip()

# Usage
llava = LLaVAModel()
image = Image.open("chart.png")
prompt = "What does this chart show? Summarize the key trends."
response = llava.generate_response(image, prompt)
print(response)
```

### Training Pipeline

**Stage 1: Feature Alignment (Concept Captions)**
- Freeze vision encoder and LLM
- Train projection layer only
- Data: CC3M (595K image-text pairs)

**Stage 2: Visual Instruction Tuning**
- Freeze vision encoder
- Fine-tune projection + LLM with LoRA
- Data: GPT-4 generated visual instructions (158K)

```python
# LLaVA training configuration
from peft import LoraConfig, get_peft_model

# LoRA for LLM
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply to LLM part only
model.language_model = get_peft_model(model.language_model, lora_config)

# Training: freeze vision tower, train projection + LoRA
for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True
```

## Cross-Modal Attention

### Implementing Cross-Attention

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Q from text, K/V from vision
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = self.d_head ** -0.5
    
    def forward(self, text_features, vision_features, vision_mask=None):
        """
        text_features: (batch, text_len, d_model)
        vision_features: (batch, vision_len, d_model)
        """
        batch_size = text_features.size(0)
        
        # Project
        Q = self.q_proj(text_features)
        K = self.k_proj(vision_features)
        V = self.v_proj(vision_features)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply vision mask
        if vision_mask is not None:
            vision_mask = vision_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(vision_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(out)

# Gated cross-attention (Flamingo-style)
class GatedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm_text = nn.LayerNorm(d_model)
        self.norm_vision = nn.LayerNorm(d_model)
    
    def forward(self, text_features, vision_features, vision_mask=None):
        # Normalize
        text_norm = self.norm_text(text_features)
        vision_norm = self.norm_vision(vision_features)
        
        # Cross-attention
        attended = self.cross_attn(text_norm, vision_norm, vision_mask)
        
        # Gated residual
        output = text_features + torch.tanh(self.gate) * attended
        
        return output
```

## Visual Question Answering (VQA)

### Implementation

```python
class VQAModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, fusion_dim=512, num_answers=3000):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Projection to common space
        self.vision_proj = nn.Linear(vision_encoder.config.hidden_size, fusion_dim)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, fusion_dim)
        
        # Fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.classifier = nn.Linear(fusion_dim, num_answers)
    
    def forward(self, image, question):
        # Encode image
        vision_outputs = self.vision_encoder(image)
        vision_features = vision_outputs.last_hidden_state[:, 0]  # [CLS] token
        vision_features = self.vision_proj(vision_features)
        
        # Encode question
        text_outputs = self.text_encoder(**question)
        text_features = text_outputs.last_hidden_state[:, 0]  # [CLS] token
        text_features = self.text_proj(text_features)
        
        # Fusion
        fused = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion(fused)
        
        # Classify
        logits = self.classifier(fused)
        return logits

# Training
def train_vqa(model, train_loader, optimizer, criterion):
    for batch in train_loader:
        images = batch['images']
        questions = batch['questions']
        answers = batch['answers']
        
        logits = model(images, questions)
        loss = criterion(logits, answers)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Image Captioning

### Encoder-Decoder Architecture

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_encoder, decoder, vocab_size):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.decoder = decoder
        
        # Projection from vision to decoder dimension
        self.projection = nn.Linear(
            vision_encoder.config.hidden_size,
            decoder.config.hidden_size
        )
        
        self.vocab_projection = nn.Linear(
            decoder.config.hidden_size,
            vocab_size
        )
    
    def forward(self, images, captions=None, max_length=50):
        # Encode image
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state[:, 0]
        image_embeds = self.projection(image_features)
        
        if captions is not None:
            # Training: teacher forcing
            decoder_inputs = captions[:, :-1]
            decoder_outputs = self.decoder(
                inputs_embeds=image_embeds.unsqueeze(1),
                decoder_input_ids=decoder_inputs
            )
            logits = self.vocab_projection(decoder_outputs.last_hidden_state)
            return logits
        else:
            # Inference: autoregressive generation
            generated = torch.zeros(images.size(0), 1, dtype=torch.long)
            
            for _ in range(max_length):
                decoder_outputs = self.decoder(
                    inputs_embeds=image_embeds.unsqueeze(1),
                    decoder_input_ids=generated
                )
                logits = self.vocab_projection(decoder_outputs.last_hidden_state[:, -1])
                next_token = logits.argmax(dim=-1)
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            return generated
```

## Model Comparison

| Model | Architecture | Training | Strengths |
|-------|-------------|----------|-----------|
| **CLIP** | Dual-Encoder | Contrastive | Zero-shot, retrieval |
| **BLIP** | Fusion + Decoder | Bootstrap | Captioning, VQA |
| **BLIP-2** | Q-Former + LLM | 2-stage | Reasoning, efficiency |
| **LLaVA** | Vision Encoder + LLM | Instruction tuning | General assistant |
| **Flamingo** | Perceiver + Frozen LLM | Cross-attention | Few-shot learning |
| **GPT-4V** | Unknown | Unknown | SOTA capabilities |

## Best Practices

### 1. Vision Encoder Selection

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **General purpose** | ViT-L/14 | Good balance |
| **High resolution** | ViT-H/14 | Better detail |
| **Efficiency** | CLIP-ViT-B/32 | Faster |
| **Medical imaging** | Custom pre-trained | Domain specific |

### 2. Training Strategies

```python
# Multi-task learning
class MultitaskVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = load_vision_encoder()
        self.text_encoder = load_text_encoder()
        self.fusion = CrossAttention(d_model=768, num_heads=12)
        
        # Task heads
        self.caption_head = nn.Linear(768, vocab_size)
        self.vqa_head = nn.Linear(768, num_answers)
        self.retrieval_head = nn.Linear(768, embed_dim)
    
    def forward(self, images, text, task):
        vision_feats = self.vision_encoder(images)
        text_feats = self.text_encoder(text)
        fused = self.fusion(text_feats, vision_feats)
        
        if task == "caption":
            return self.caption_head(fused)
        elif task == "vqa":
            return self.vqa_head(fused)
        elif task == "retrieval":
            return self.retrieval_head(fused)
```

### 3. Data Augmentation

```python
# Visual augmentation for robustness
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 4. Evaluation Metrics

| Task | Metrics |
|------|---------|
| **Image Captioning** | BLEU, METEOR, CIDEr, SPICE |
| **VQA** | Accuracy |
| **Retrieval** | Recall@K, mAP |
| **Generation** | Perplexity, Human evaluation |

## Common Pitfalls

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Modality gap** | Vision/text embeddings not aligned | Better contrastive training, temperature tuning |
| **Hallucination** | Generated text mentions objects not in image | Grounded training, object detection constraints |
| **Language bias** | Model ignores visual information | Balanced vision-language loss |
| **Resolution limits** | Fine details missed | Higher resolution encoders, patch selection |

---

**Previous**: [Alignment](../alignment/README.md) | **Next**: [RAG Foundations](../../05-RAG-Systems/rag-foundations/README.md)

*Note: Phase 5 (RAG Systems) and Phase 6 (MLOps Production) contain complete production-ready content already established in the repository.*