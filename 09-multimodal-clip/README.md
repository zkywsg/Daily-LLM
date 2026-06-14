# 多模态对齐 (CLIP / 跨模态)

> **让图像和文本在同一向量空间里对齐,跨模态从"专门设计的接口"变成"自然语言驱动的通用能力"。**

## 一句话定位

这家族解决的是视觉和语言这两条独立发展的深度学习主线如何"互通"的根本问题。2020 年之前,视觉和 NLP 各自有自己的模型(CNN vs Transformer)、各自的训练数据(ImageNet vs WebText)、各自的任务(分类 vs LM),**跨模态任务需要专门设计**——image captioning 要训 CNN+RNN+attention 的复杂 pipeline,VQA 要堆 vision feature + question encoder + fusion module。2021 年 OpenAI 的 CLIP 给出了截然不同的方案——**用 4 亿对网络图文数据,通过对比学习把图像和文本编码到同一向量空间**,让"图片是关于什么的"和"文本描述的是什么"用余弦相似度直接比较。CLIP 不只是 zero-shot 分类(给一组类别 prompt,选相似度最高的)的开关键,更是后续多模态生成 / VLM / 视觉 agent 的**对齐基座**——Stable Diffusion 用它做 text encoder、DALL-E 2 用它做 prior、几乎所有 VLM 都用它做视觉特征提取。2022-2023 年这家族出现两个重要方向:**BLIP** 系列在 CLIP 基础上加入生成能力(image captioning, VQA);**Flamingo** 给出"冻结大 LLM + 加视觉接口"的范式;**LLaVA** 用 instruction tuning 把 GPT-4V 风格的视觉助手开源化。今天的 GPT-4V / Claude 3 / Gemini 1.5 多模态能力都建立在这条家族的演化上。

## 概念本身

跨模态对齐的核心是**让不同模态的语义内容在同一向量空间里可比较**。具体三步:

**1. 模态独立编码器** —— 图像用 [ViT](../08-vit/01-vit.md) 编码成向量,文本用 [Transformer](../05-transformer/01-transformer.md) 编码成向量,**两个编码器各自独立**,只在最后输出层意义上对齐

**2. 对比学习目标** —— 给定 batch 里 N 对真实图文对 `(x_i, y_i)`,让 `cos(image_emb_i, text_emb_i)` 高,`cos(image_emb_i, text_emb_j) for j≠i` 低。这是 InfoNCE 损失:

$$
\mathcal{L} = -\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(x_i, y_i) / \tau)}{\sum_j \exp(\text{sim}(x_i, y_j) / \tau)}
$$

**3. 大规模图文数据** —— 互联网上有海量"图 + 描述"对(网页 alt-text、社交媒体配文、商品图说明),不需要任务标注,可以 scale 到 4 亿 / 10 亿 / 50 亿规模

这三件事一起做,CLIP 学到的不只是"图文匹配"——而是**用语言定义视觉概念的通用表征**。"红色的圆"在视觉编码里和文本编码里都有清晰位置,可以用文本任意组合查询任何概念。

这家族围绕几条主线演化:

- **从匹配到生成**:CLIP(对比)→ BLIP(对比 + 生成 + 匹配三任务联合)
- **从专用到通用**:CLIP 时代每个任务训一个模型 → Flamingo 的"冻结大 LLM + 视觉适配"少样本范式
- **开源化**:闭源前沿(GPT-4V)→ LLaVA 等开源 VLM
- **多模态融合的深度**:从"双塔晚融合"到"早融合 single Transformer"(SD3 的 MM-DiT)

CLIP / 跨模态家族在 2021-2024 是 AI 应用爆发的根基——文生图、视觉 agent、AI 助手的多模态能力都源于这条线。理解这家族 = 理解今天 AI 系统怎么"看"世界。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2021 | **CLIP** | 4 亿对图文对比学习,图像和文本编码到同一空间,zero-shot 分类匹配监督 SOTA;通用视觉表征基座 | 跨模态任务需要专门设计 pipeline;视觉表征只能用 ImageNet 类别监督学 |
| 2022 | **BLIP / BLIP-2** | 对比 + 生成 + 匹配三任务联合预训练;BLIP-2 用 Q-Former 桥接冻结视觉编码器和冻结 LLM | CLIP 只能匹配不能生成 caption / 答 VQA |
| 2022 | **Flamingo** | 冻结大 LLM(Chinchilla 70B)+ Perceiver Resampler 视觉适配 + cross-attention 注入,8 例 in-context 学新视觉任务 | 多模态训练成本极高;每个任务从头训不可行 |
| 2023 | **LLaVA** | Visual instruction tuning:用 GPT-4 生成视觉指令数据,把 CLIP 视觉特征接到 LLaMA,开源 VLM 范式定型 | GPT-4V 闭源;开源社区缺少能用的视觉助手 |

## 依赖与延伸

**前置(foundations):**
- [../08-vit/01-vit.md](../08-vit/01-vit.md) —— 多数 CLIP 系模型的图像编码器是 ViT
- [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) —— CLIP 的文本编码器是 Transformer,BLIP 借鉴 BERT 思想
- [../07-gpt-scaling/03-gpt3.md](../07-gpt-scaling/03-gpt3.md) —— Flamingo / LLaVA 的 LLM backbone
- [../12-rlhf-alignment/02-instructgpt.md](../12-rlhf-alignment/02-instructgpt.md) —— LLaVA 用类似 instruction tuning 思路

**通向哪些家族:**
- [../10-diffusion/](../10-diffusion/) —— Stable Diffusion 用 CLIP text encoder
- [../14-rag-agent/](../14-rag-agent/) —— 多模态 RAG / 视觉 agent 依赖 CLIP-style 检索
- [../15-reasoning-o1-r1/](../15-reasoning-o1-r1/) —— 多模态推理 / 视觉 CoT 建立在 VLM 之上
