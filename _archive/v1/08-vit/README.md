# 视觉 Transformer (ViT)

> **把 Transformer 从 NLP 整体搬到视觉:图像切成 patch 当 token,attention 替代卷积,在足够大数据上击败 CNN。**

## 一句话定位

这家族解决的是视觉领域 2020 年前后的一个反复被问的问题——**Transformer 能不能不靠 CNN 帮忙,直接处理图像?** 在 ViT 之前,视觉 Transformer 都是混合架构(CNN backbone 提取特征 + Transformer 做高层)——因为 attention 的计算复杂度 O(N²) 对图像的几万像素直接不可行,且大家相信"卷积的归纳偏置(locality + translation equivariance)对视觉是必需的"。2020 年 10 月 Google 的 ViT 用一个极简方案打破了这个共识——**把图像切成 16×16 的 patch,每个 patch 拉平当一个 token,然后用纯 Transformer encoder 处理**。在 JFT-300M 这个巨大私有数据集上预训练后,ViT 在 ImageNet 上击败了同等规模的 ResNet/EfficientNet。这一结果直接证明:**视觉的归纳偏置不是必需的,数据足够多时 attention 能从零学到**——这是从 CNN 时代到 Transformer 时代的范式转折。这家族要回答的问题是:**ViT 之后视觉 Transformer 是怎么演化的——数据效率(DeiT)、层级表征(Swin)、跨任务统一(DiT)**。

## 概念本身

视觉 Transformer 的核心是**把图像离散化成 token 序列**,然后套用 NLP 那套 Transformer 流程。具体三步:

**1. Patch embedding**——把 `H × W × 3` 的图像切成 `N × N` 的 patch(典型 N=16),每个 patch 拉平成 `N²·3` 维向量,过一个 linear 投影到 `d_model` 维。一张 `224 × 224` 图像切成 `14 × 14 = 196` 个 patch,加上 `[CLS]` token 共 197 个 token

**2. 位置编码 + Transformer encoder**——给每个 patch token 加 learned position embedding(标记空间位置),整个序列过标准 Transformer encoder 堆叠(典型 12-32 层)

**3. 分类头**——`[CLS]` token 的最终 hidden state 接 linear classifier 做 ImageNet 1000 分类(或其他下游任务)

```
图像 [224, 224, 3]
  ↓ Patchify(16×16)
196 个 patch,每个 768 维
  ↓ + [CLS] token + Position embedding
197 × 768 序列
  ↓ Transformer encoder × 12-32 层
197 × 768 上下文表征
  ↓ 取 [CLS] 位置 + linear classifier
1000 类 logits
```

这条路线的几个关键 trade-off:

**1. 无视觉归纳偏置**——CNN 的"局部连接 + 参数共享"对像素相邻这件事是硬编码的;Transformer 没有这一假设,所有 patch 之间都是 fully connected。这让 Transformer 在**小数据(< 1M 图像)上不如 CNN**——没有合适归纳偏置,模型在小数据上学不出有效特征。但在大数据(> 100M)上反而比 CNN 强——因为 CNN 的归纳偏置在足够数据时反而成了限制

**2. attention O(N²) 复杂度**——14×14=196 个 patch 时 196² = 38K 个 attention 分数,可控;但高分辨率图像(512×512 = 1024 patches, N² = 1M)就不可行。这是 [Swin Transformer](03-swin.md) 用 windowed attention 解决的问题

**3. 跨任务统一**——CNN 时代不同任务(分类 / 检测 / 分割)用不同 backbone;Transformer 可以用同一 backbone 处理多种任务,只需换 head。这一点是 ViT 时代最重要的工程价值之一

这家族的演化主线:

- **数据效率**:[DeiT](02-deit.md) 用蒸馏 + 强数据增强让 ViT 不依赖 JFT-300M,ImageNet-1K 训练也能 work
- **层级化 + 高分辨率**:[Swin](03-swin.md) 用 shifted window attention,把复杂度从 O(N²) 降到 O(N),detection / segmentation 上 SOTA
- **跨任务**:[DiT](04-dit.md) 把 ViT 思想搬到 diffusion 生成,Sora / Stable Diffusion 3 的基座

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2020 | **ViT** | 把图像切成 16×16 patch 当 token + 纯 Transformer encoder,在 JFT-300M 上预训练后在 ImageNet 上击败 ResNet,证明视觉归纳偏置不是必需的 | 视觉 Transformer 都是混合架构;社区相信卷积归纳偏置对视觉是必需的 |
| 2021 | **DeiT** | 用 distillation token + 强增强 + AdamW,让 ViT 在 ImageNet-1K(1.3M 图像)上从零训练能击败 ResNet,不再依赖 JFT-300M | ViT 在小数据(< 100M)上不如 CNN,学界很难复现(没有 JFT 访问权限) |
| 2021 | **Swin Transformer** | shifted window attention,把 O(N²) 降到 O(N) 支持高分辨率;层级化特征图;detection / segmentation 全面 SOTA | ViT 只能做分类,不能直接套到 detection / segmentation;高分辨率(>384)算力爆炸 |
| 2022 | **DiT** | 把 U-Net 替换成 Transformer 做 diffusion,scaling 显著更好,Stable Diffusion 3 / Sora 的基座 | Diffusion 模型一直用 U-Net,无法享受 Transformer 的 scaling law |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— ViT 直接用 Transformer encoder
- [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) —— ViT 的 [CLS] token 和分类范式都借鉴 BERT
- [../01-cnn/05-resnet.md](../01-cnn/05-resnet.md) —— ViT 时代之前视觉的标准 baseline
- [../01-cnn/08-convnext.md](../01-cnn/08-convnext.md) —— 2022 用现代 recipe 把 ResNet 调回到 ViT 水平的"CNN 反击"

**通向哪些家族:**
- [../09-multimodal-clip/](../09-multimodal-clip/) —— CLIP 的图像编码器是 ViT
- [../10-diffusion/](../10-diffusion/) —— DiT 把 ViT 思想用到 diffusion 生成
- [../14-rag-agent/](../14-rag-agent/) —— 多模态 RAG 的视觉理解依赖 ViT 系编码器
