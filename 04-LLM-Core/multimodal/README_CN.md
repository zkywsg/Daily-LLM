# 多模态模型

[English](README.md) | [中文](README_CN.md)

## 概述

多模态模型处理和生成跨多种模态的内容 (文本、图像、音频、视频)。本指南专注于将计算机视觉与语言理解结合的视觉-语言模型。

## 视觉-语言架构模式

### 1. 双编码器 (Dual-Encoder, 对比式)

**每种模态使用单独的编码器,在共享空间中对齐:**

```
Image → Image Encoder → Image Embedding
                              ↓ (对比损失)
Text  → Text Encoder  → Text Embedding
```

**示例**: CLIP, ALIGN

### 2. 融合编码器

**模态的早期/中间融合:**

```
Image → Vision Encoder ─┐
                        ├→ Fusion Layers → Decoder → Output
Text  → Text Encoder  ─┘
```

**示例**: VisualBERT, ViLBERT, UNITER

### 3. 带视觉的编码器-解码器

**视觉编码器 + 文本解码器 (自回归):**

```
Image → Vision Encoder → Projector → Text Decoder → Caption/Response
```

**示例**: BLIP, BLIP-2, LLaVA, GPT-4V

### 4. Flamingo 风格 (Perceiver Resampler)

**冻结 LLM 中的门控交叉注意力:**

```
Image Patches → Perceiver Resampler → Visual Tokens
                                           ↓
Text Tokens → Frozen LLM with Gated Cross-Attention → Output
```

**示例**: Flamingo, OpenFlamingo, IDEFICS

## CLIP (对比式语言-图像预训练)

### 架构

| Component | Specification |
|-----------|--------------|
| Image Encoder | ViT 或 ResNet |
| Text Encoder | Transformer |
| Projection | 线性层到共享空间 |
| Training | 在 (图像,文本) 对上的对比学习 |

### 对比学习

**目标**: 最大化匹配对的相似度,最小化不匹配对的相似度

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
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.T) / self.temperature

        # 标签: 对角元素是正样本对
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # 对称损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

# 使用 HuggingFace CLIP
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图像-文本相似度
image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a car"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# 计算相似度
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)  # "a photo of a cat" 最高
```

### 零样本分类

```python
def zero_shot_classify(image, class_names, model, processor):
    """不训练即可分类图像"""
    # 创建提示词
    texts = [f"a photo of a {name}" for name in class_names]

    # 处理
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # 获取预测
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=0)

    # 返回前几个预测
    top_probs, top_indices = probs.topk(5)
    return [(class_names[i], p.item()) for i, p in zip(top_indices, top_probs)]

# 使用
results = zero_shot_classify(image, ["cat", "dog", "bird", "fish"], model, processor)
```

### 图像检索

```python
def build_image_index(images, model, processor):
    """构建可搜索的图像索引"""
    image_features = []

    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
        image_features.append(features)

    return torch.cat(image_features, dim=0)

def search_images(query_text, image_index, model, processor, top_k=5):
    """通过文本查询搜索图像"""
    # 编码查询
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)

    # 计算相似度
    similarities = torch.matmul(text_features, image_index.T)

    # 获取前几个匹配
    top_scores, top_indices = similarities[0].topk(top_k)
    return top_indices.tolist(), top_scores.tolist()
```

## BLIP (自举语言-图像预训练)

### 统一架构

BLIP 统一了视觉-语言理解和生成:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# 加载 BLIP 模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 图像描述
image = Image.open("beach.jpg")
inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs, max_length=50)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)  # "a group of people walking on a beach with surfboards"
```

### 自举训练

BLIP 使用标题过滤器自举训练数据:

```
原始网络数据 → 嘈杂的 (图像,文本) 对
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
            Caption Filter (移除嘈杂标题)
                   ↓
            Bootstrapped Training Data
```

## BLIP-2: 高效视觉-语言预训练

### Q-Former (Querying Transformer, 查询 Transformer)

**桥接冻结的图像编码器和冻结的 LLM:**

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 加载 BLIP-2 (OPT-2.7B)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
)

# 视觉问答
image = Image.open("kitchen.jpg")
prompt = "Question: What appliance is in image? Answer:"

inputs = processor(image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)  # "refrigerator"
```

### 两阶段训练

**阶段 1: 视觉-语言表示学习**
- 使用冻结的图像编码器训练 Q-Former
- 使用 ITC、ITM 和图像锚定文本生成目标

**阶段 2: 视觉到语言生成学习**
- 将 Q-Former 连接到冻结的 LLM (OPT 或 Flan-T5)
- 训练以根据视觉查询生成文本

## LLaVA (Large Language and Vision Assistant, 大语言和视觉助手)

### 架构

**视觉编码器 + 投影层 + LLM:**

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
        """生成对图像 + 提示词的文本响应"""
        # 使用图像 token 格式化提示词
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # 生成
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        # 提取助手响应
        return response.split("ASSISTANT:")[-1].strip()

# 使用
llava = LLaVAModel()
image = Image.open("chart.png")
prompt = "What does this chart show? Summarize key trends."
response = llava.generate_response(image, prompt)
print(response)
```

### 训练流水线

**阶段 1: 特征对齐 (Concept Captions)**
- 冻结视觉编码器和 LLM
- 仅训练投影层
- 数据: CC3M (595K 图像-文本对)

**阶段 2: 视觉指令微调**
- 冻结视觉编码器
- 使用 LoRA 微调投影 + LLM
- 数据: GPT-4 生成的视觉指令 (158K)

```python
# LLaVA 训练配置
from peft import LoraConfig, get_peft_model

# 用于 LLM 的 LoRA
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 仅应用于 LLM 部分
model.language_model = get_peft_model(model.language_model, lora_config)

# 训练: 冻结视觉塔,训练投影 + LoRA
for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True
```

## 跨模态注意力

### 实现交叉注意力

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q 来自文本,K/V 来自视觉
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

        # 投影
        Q = self.q_proj(text_features)
        K = self.k_proj(vision_features)
        V = self.v_proj(vision_features)

        # 为多头重塑
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用视觉掩码
        if vision_mask is not None:
            vision_mask = vision_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(vision_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # 将注意力应用于值
        out = torch.matmul(attn, V)

        # 拼接头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out)

# 门控交叉注意力 (Flamingo 风格)
class GatedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm_text = nn.LayerNorm(d_model)
        self.norm_vision = nn.LayerNorm(d_model)

    def forward(self, text_features, vision_features, vision_mask=None):
        # 归一化
        text_norm = self.norm_text(text_features)
        vision_norm = self.norm_vision(vision_features)

        # 交叉注意力
        attended = self.cross_attn(text_norm, vision_norm, vision_mask)

        # 门控残差
        output = text_features + torch.tanh(self.gate) * attended

        return output
```

## 视觉问答 (Visual Question Answering, VQA)

### 实现

```python
class VQAModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, fusion_dim=512, num_answers=3000):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # 投影到公共空间
        self.vision_proj = nn.Linear(vision_encoder.config.hidden_size, fusion_dim)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, fusion_dim)

        # 融合和分类器
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        self.classifier = nn.Linear(fusion_dim, num_answers)

    def forward(self, image, question):
        # 编码图像
        vision_outputs = self.vision_encoder(image)
        vision_features = vision_outputs.last_hidden_state[:, 0]  # [CLS] token
        vision_features = self.vision_proj(vision_features)

        # 编码问题
        text_outputs = self.text_encoder(**question)
        text_features = text_outputs.last_hidden_state[:, 0]  # [CLS] token
        text_features = self.text_proj(text_features)

        # 融合
        fused = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion(fused)

        # 分类
        logits = self.classifier(fused)
        return logits

# 训练
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

## 图像描述

### 编码器-解码器架构

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_encoder, decoder, vocab_size):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.decoder = decoder

        # 从视觉到解码器维度的投影
        self.projection = nn.Linear(
            vision_encoder.config.hidden_size,
            decoder.config.hidden_size
        )

        self.vocab_projection = nn.Linear(
            decoder.config.hidden_size,
            vocab_size
        )

    def forward(self, images, captions=None, max_length=50):
        # 编码图像
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state[:, 0]
        image_embeds = self.projection(image_features)

        if captions is not None:
            # 训练: teacher forcing
            decoder_inputs = captions[:, :-1]
            decoder_outputs = self.decoder(
                inputs_embeds=image_embeds.unsqueeze(1),
                decoder_input_ids=decoder_inputs
            )
            logits = self.vocab_projection(decoder_outputs.last_hidden_state)
            return logits
        else:
            # 推理: 自回归生成
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

## 模型对比

| Model | Architecture | Training | Strengths |
|-------|-------------|----------|-----------|
| **CLIP** | 双编码器 | 对比式 | 零样本、检索 |
| **BLIP** | 融合 + 解码器 | 自举 | 描述、VQA |
| **BLIP-2** | Q-Former + LLM | 2 阶段 | 推理、效率 |
| **LLaVA** | 视觉编码器 + LLM | 指令微调 | 通用助手 |
| **Flamingo** | Perceiver + 冻结 LLM | 交叉注意力 | 少样本学习 |
| **GPT-4V** | 未知 | 未知 | SOTA 能力 |

## 最佳实践

### 1. 视觉编码器选择

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **通用** | ViT-L/14 | 良好的平衡 |
| **高分辨率** | ViT-H/14 | 更好的细节 |
| **效率** | CLIP-ViT-B/32 | 更快 |
| **医学成像** | 定制预训练 | 领域特定 |

### 2. 训练策略

```python
# 多任务学习
class MultitaskVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = load_vision_encoder()
        self.text_encoder = load_text_encoder()
        self.fusion = CrossAttention(d_model=768, num_heads=12)

        # 任务头
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

### 3. 数据增强

```python
# 用于鲁棒性的视觉增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 4. 评估指标

| Task | Metrics |
|------|---------|
| **图像描述** | BLEU, METEOR, CIDEr, SPICE |
| **VQA** | 准确率 |
| **检索** | Recall@K, mAP |
| **生成** | 困惑度、人工评估 |

## 常见陷阱

| Issue | Symptom | Solution |
|-------|---------|----------|
| **模态差距** | 视觉/文本嵌入不对齐 | 更好的对比训练、温度调优 |
| **幻觉** | 生成的文本提到图像中不存在的物体 | 锚定训练、目标检测约束 |
| **语言偏差** | 模型忽略视觉信息 | 平衡的视觉-语言损失 |
| **分辨率限制** | 错过细节 | 更高分辨率的编码器、补丁选择 |

---

**上一节**: [对齐](../alignment/README.md) | **下一节**: [RAG 基础](../../05-RAG-Systems/rag-foundations/README.md)

*注: 阶段 5 (RAG 系统) 和阶段 6 (MLOps 生产) 包含仓库中已建立的完整生产就绪内容。*
