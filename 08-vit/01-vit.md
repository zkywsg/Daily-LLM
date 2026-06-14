---
name: "ViT"
year: 2020
family: "08-vit"
order: 1
paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", "Dirk Weissenborn", "Xiaohua Zhai", "Thomas Unterthiner", "Mostafa Dehghani", "et al."]
key_idea: "把图像切成 16×16 的 patch 当 token,用纯 Transformer encoder 处理,在 JFT-300M 上预训练后击败 CNN,证明视觉归纳偏置不是必需的"
---

## 前作进展

到 2020 年中,CNN 在视觉领域已经主导了 8 年([AlexNet](../01-cnn/02-alexnet.md) 2012 之后),ResNet / EfficientNet 是 ImageNet 上的事实 SOTA。NLP 那边 [Transformer](../05-transformer/01-transformer.md) 一统天下,但视觉社区对"能不能用 Transformer 替代 CNN"持有几乎共识性的否定:

**为什么社区相信 CNN 不可替代?**

**1. attention 的 O(N²) 复杂度**——对一张 `224 × 224` 图像直接做 pixel-level attention 是 `(224 × 224)² = 2.5 × 10⁹` 次内积,不可行。CNN 的局部连接是绕开这个问题的天然解法

**2. CNN 的归纳偏置被认为"必需"**——卷积内置了三件事:**局部连接**(像素邻居更相关)、**参数共享**(同一滤波器在空间上扫)、**translation equivariance**(平移不变性)。社区相信这些先验对视觉是结构性必需,Transformer 没有就学不到

**3. 之前的混合架构尝试效果一般**——2018-2019 年的 ViT 前作(Bello *Attention Augmented Convolutions*, Ramachandran *Stand-Alone Self-Attention*)都是"CNN backbone + 部分 attention 替换 conv",效果有改进但不显著,且没有展示出"纯 Transformer > CNN"的论据

Google Brain 团队 2020 年 10 月发表 *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*(ViT)做了完全相反的尝试:**完全不用卷积,把图像切成 patch 直接喂给 Transformer encoder**。结果令人惊讶——**在 JFT-300M(Google 内部 3 亿图像数据集)上预训练后,ViT 在 ImageNet 上击败了同等规模的 ResNet,且训练算力少 4×**。

这一结果直接打破了"CNN 不可替代"的共识。它的关键启示:**视觉归纳偏置不是必需的,只要数据足够多,Transformer 能从零学到**。这一观点在 2021 年的 CLIP / DALL-E、2022 年的 Stable Diffusion、2023 年的 DiT 上反复被验证,**Transformer 在 2022 之后几乎全面替代 CNN 成为视觉默认骨干**——只有需要极低算力的边缘场景(移动端)还在用 CNN。

## 核心思想:Image as Patch Sequence

ViT 的核心是**把图像处理转化为序列处理**。具体三步:

```mermaid
graph LR
    img["Image [224,224,3]"]:::input --> patch["Patchify 16×16<br/>→ 196 patches"]:::compute
    patch --> proj["Linear proj<br/>→ [196, 768]"]:::compute
    proj --> cls["+ [CLS] token<br/>+ Position emb<br/>→ [197, 768]"]:::compute
    cls --> enc["Transformer encoder<br/>× 12 层"]:::compute
    enc --> head["[CLS] → MLP head<br/>→ 1000 classes"]:::output

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```

*图 1:ViT 完整 pipeline——图像 → patchify → linear embed → 加 [CLS] + 位置编码 → Transformer encoder → 分类。整个流程没有任何卷积。*

**Step 1: Patchify** —— 把 `H × W × 3` 图像切成 `P × P` 的不重叠 patch。ViT-B/16 用 `P=16`,所以 `224 × 224` 图像切成 `(224/16)² = 14 × 14 = 196` 个 patch。每个 patch 是 `16 × 16 × 3 = 768` 维向量(拉平)。

**Step 2: Linear projection** —— 每个 patch 过一个 `768 → 768` 的 linear 层(其实就是一个 `16×16` stride 16 的卷积,但 ViT 论文坚持称之为 "patch embedding"以强调"无卷积"的概念)。输出 `[196, 768]` 序列。

**Step 3: 加 [CLS] + position embedding** —— 序列前缀加一个可学习的 `[CLS]` token(同 BERT),然后加 197 维的 learned position embedding。最终得到 `[197, 768]` token 序列。

**Step 4: Transformer encoder** —— 12 层标准 Transformer encoder(同 [BERT-base](../06-bert-family/01-bert.md) 配置,只是输入分布不同)。Self-attention 不带 causal mask,完全双向。

**Step 5: 分类头** —— 取 `[CLS]` 位置的最终 hidden state,过一个 MLP head 输出 1000 类 logits。

整个 pipeline 没有任何专门的视觉模块——**ViT 就是 BERT 的视觉版**,差异只在输入预处理(patchify 替代 tokenization)。

## 模型规格

ViT 的命名遵循 `ViT-<size>/<patch_size>` 模式:

| 模型 | 层数 | d_model | h | d_ff | 参数 |
|------|------|------|------|------|------|
| ViT-B/16 | 12 | 768 | 12 | 3072 | 86M |
| ViT-L/16 | 24 | 1024 | 16 | 4096 | 307M |
| ViT-H/14 | 32 | 1280 | 16 | 5120 | 632M |

Patch size 越小(`14 < 16 < 32`),序列越长、attention 计算量越大,但表征更细粒度。ViT-H/14 是论文里最强配置——14×14 patch(`224/14=16` patches each direction,共 256 个),配合 32 层 transformer 达到 ImageNet SOTA。

## 数据规模的关键

ViT 论文最重要的发现是**性能-数据规模的关系**(论文 Figure 3):

| 预训练数据集 | 规模 | ViT-L/16 ImageNet acc | ResNet-152 ImageNet acc |
|------|------|------|------|
| ImageNet-1K | 1.3M | 76.5 | **77.8**(CNN 胜) |
| ImageNet-21K | 14M | 84.0 | 82.7(平手) |
| **JFT-300M** | **300M** | **87.8**(ViT 胜) | 86.2 |

观察:**ViT 在小数据(1.3M)上不如 ResNet,在中等数据(14M)上持平,在大数据(300M)上明显胜出**。这一现象的物理解释是:

- **CNN 的归纳偏置是"免费的训练数据"**——locality 和 translation equivariance 这两条先验等于告诉模型"邻居相关 + 平移不变",省去了从数据中学这些事的成本。小数据时 CNN 利用归纳偏置高效收敛
- **Transformer 必须从数据中学到这些先验**——大数据足够时,Transformer 从数据中学到的视觉表征**比 CNN 硬编码的归纳偏置更灵活**(可以学到非平移不变性的模式,如"图像中央更重要")

这一发现的方法论意义:**归纳偏置不是免费午餐,在足够数据时它可能是约束而不是优势**。LeCun 2022 称此为 "the bitter lesson of vision" ——重复了 2012 年 [AlexNet](../01-cnn/02-alexnet.md) 打败 SIFT+SVM 时的故事:**学到的特征 > 设计的特征**。

但 JFT-300M 是 Google 私有数据集,学界没访问权限。**这一限制让 ViT 在发表后近半年内学界无法复现**,直到 [DeiT](02-deit.md)(2021)用 ImageNet-1K + 强增强 + 蒸馏在小数据上让 ViT 也 work,才让 ViT 真正普及到学界。

## 性能数据

ViT 在 ImageNet 上的成绩(论文 Table 2):

| 模型 | 预训练 | ImageNet top-1 | ImageNet ReaL | TPUv3 训练天数 |
|------|------|------|------|------|
| ResNet-152x4(Big Transfer) | JFT-300M | 87.5 | 90.5 | **9.9** |
| EfficientNet-L2 | ImageNet | 88.5 | 90.6 | 12.3 |
| **ViT-H/14** | **JFT-300M** | **88.6** | **90.7** | **2.5** |

ViT-H 在 ImageNet 上**比 EfficientNet-L2 略好,训练时间少 5×**——这是关键的工程论据。视觉 SOTA 在 EfficientNet 时代已经卡在 88.5%,ViT 突破到 88.6+,且训练算力少得多。

ImageNet ReaL(更严格的重标注版)上 ViT 90.7 也是 SOTA。在 19 个迁移学习任务上,ViT 平均也比 BiT(Big Transfer, ResNet-152x4)略好。

## 训练细节

| 维度 | ViT-L/16 in JFT |
|------|------|
| 架构 | 24 层 Transformer encoder, d=1024, h=16, d_ff=4096, 307M 参数 |
| Patch | 16×16,224 输入 → 196 patches |
| Position embedding | learned 1D(论文比较过 2D / 相对 PE,1D learned 效果一样好) |
| 预训练数据 | JFT-300M(300M 图像,18K 类) |
| 预训练目标 | 多标签分类(因为 JFT 有多标签) |
| 优化器 | Adam(β1=0.9, β2=0.999),线性 warmup + 线性 decay |
| Learning rate | 1e-3 |
| Weight decay | 0.1 |
| Batch | 4096 |
| 预训练 epoch | 7(对 JFT-300M ≈ 2.1B 图像) |
| 预训练时长 | TPUv3-2500 cores × 30 天 |
| 微调 | ImageNet 上微调,lr 0.01, batch 512 |
| 微调时长 | 几小时(几个 epoch) |

注意一个工程细节——**预训练时模型经常 collapse**(loss 突然 NaN)。Dosovitskiy 团队报告 ViT-H 训练在 30K 步左右经常发生这种情况,需要 restart from checkpoint + lr warm restart。这是大模型预训练的常见问题,后来 Pre-LN + RMSNorm 等改动让训练稳定性显著提升。

## 关键代码

ViT 的核心实现极其简洁(基于 timm / official 简化):

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # 等价于 stride=patch_size 的 conv2d,但论文坚持称为 "patch embedding"
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.proj(x)                 # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2) # [B, 196, 768]
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        # [CLS] token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        # Transformer encoder(标准,Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu', batch_first=True, norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                          # [B, 196, 768]
        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, 768]
        x = torch.cat([cls, x], dim=1)                   # [B, 197, 768]
        x = x + self.pos_emb                             # 加位置编码
        x = self.encoder(x)                              # [B, 197, 768]
        x = self.norm(x[:, 0])                           # 只取 [CLS] 位置
        return self.head(x)
```

**注意几个工程要点**:

- **`patch_embed` 实质是 `Conv2d`**——论文坚持称"linear projection of flattened patches",但实现就是一个 stride=16 的 conv2d。这一"无视觉模块"的口号其实有点讨论价值
- **Pre-LN**(`norm_first=True`)——ViT 用 Pre-LN,深层稳定。这是从 GPT-2 之后的标准做法
- **`x[:, 0]` 取 [CLS]**——分类只用 `[CLS]` 位置的输出,其他 196 个 patch 的输出在分类任务上被丢弃(但在 detection/segmentation 等任务里会用)

## 影响 / 后续

ViT 在视觉历史的位置:**用一篇论文把视觉从 CNN 时代推进到 Transformer 时代**。具体影响:

**1. 视觉骨干全面 Transformer 化**——2020 之后视觉 SOTA 几乎全部是 Transformer 系:[Swin](03-swin.md) / [DeiT](02-deit.md) / BEiT / MAE / DINOv2 / SAM / CLIP / GPT-4V / Sora。CNN 退到边缘部署场景(移动端、嵌入式)

**2. 跨模态架构统一**——ViT 之后,视觉 + 语言用同一种架构(Transformer)。这让 CLIP(图像编码器 ViT + 文本编码器 Transformer)、DALL-E、BLIP 等多模态模型变得自然——所有模态共用同一计算框架

**3. 自监督视觉预训练复兴**——ViT 之后的 MAE(He 2021,*Masked Autoencoders Are Scalable Vision Learners*)把 [BERT](../06-bert-family/01-bert.md) 的 MLM 思想直接搬到视觉(随机遮 patch + 预测),自监督学习重新成为视觉研究热点

**4. 视觉 + scaling law**——ViT 论文展示了视觉模型的 scaling 行为(数据 + 参数 + 算力),这一观察催生了 ViT-G/14(2B 参数)、SoViT-22B 等超大视觉模型,后被 SAM / GPT-4V 等用作基座

**5. CNN "反击" 的 ConvNeXt**——2022 年 [ConvNeXt](../01-cnn/08-convnext.md) 用现代训练 recipe(大 kernel + LayerNorm + GELU + 强增强)把 ResNet 调到 ViT 水平。这说明 ViT 的胜利部分来自"更现代的训练 recipe",CNN 没死,只是 CNN 时代的训练方法过时了

ViT 留下的几个明确问题推动了后续节点:

- **小数据 work 不了**——依赖 JFT-300M,学界无法复现 → [DeiT](02-deit.md)
- **只能做分类**——detection / segmentation 等任务的层级特征图怎么搞? → [Swin](03-swin.md)
- **高分辨率不可行**——256+ patches 后 attention O(N²) 爆炸 → Swin 的 windowed attention
- **生成任务上 U-Net 仍主导**——diffusion 用 ViT 替代 U-Net 可行吗? → [DiT](04-dit.md)

→ [02-deit.md](02-deit.md) · 数据效率版,让 ViT 不依赖 JFT
→ [03-swin.md](03-swin.md) · 层级化 + windowed attention,支持 detection / segmentation
→ [04-dit.md](04-dit.md) · 把 ViT 用到 diffusion 生成,Sora 的基座
→ [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) · ViT 直接借鉴 [CLS] + 双向 encoder + 分类头
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · 父结构
→ [../01-cnn/08-convnext.md](../01-cnn/08-convnext.md) · CNN 的"现代化反击",回应 ViT 的胜利
→ [../09-multimodal-clip/](../09-multimodal-clip/) · CLIP 的图像编码器是 ViT
