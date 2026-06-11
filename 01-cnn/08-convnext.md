---
name: "ConvNeXt"
year: 2022
family: "01-cnn"
order: 8
paper: "A ConvNet for the 2020s"
authors: ["Zhuang Liu", "Hanzi Mao", "Chao-Yuan Wu", "Christoph Feichtenhofer", "Trevor Darrell", "Saining Xie"]
key_idea: "把 ViT 的所有现代化设计选择（大 kernel·LayerNorm·GELU·强增强）逐项搬回 ResNet，CNN 反超 ViT"
---

# ConvNeXt (2022)

## 前作进展

2020 年 [ViT](../08-vit/) 把 Transformer 直接应用到视觉，用纯注意力 + 大规模预训练把 ImageNet Top-1 提升至 85% 以上；Swin Transformer 在 2021 年又提出分层 / 滑窗注意力，在 detection / segmentation 上取代 CNN 的主流地位。两年时间内，视觉社区的关注点从"CNN 是默认 backbone"转向"CNN 是否会被 Transformer 替代"——CVPR 2021 上多数 backbone 论文转向 Transformer 变种。

但一个值得讨论的问题是：ViT 和 Swin 的优势中，有多少来自**注意力机制本身**，又有多少来自**和注意力一同引入的现代化设计组合**？ViT 用 LayerNorm 不用 BatchNorm、用 GELU 不用 ReLU、训练用 AdamW + Mixup + CutMix + RandAugment + Stochastic Depth + Label Smoothing 等组件，[ResNet](05-resnet.md) 2015 年的训练 recipe（SGD + 简单增强 + Dropout）则均未使用。**这是否构成公平的对比**？

Saining Xie（[ResNeXt](05-resnet.md) 一作）和 Zhuang Liu（[DenseNet](06-densenet.md) 一作）选择对这一问题进行系统对比。他们提的研究问题较为直接：**不引入任何新算子**，从 ResNet-50（76.1% Top-1）出发，把 Swin Transformer 用到的每一项现代化设计逐条引入，每引入一项就在 ImageNet 上测试精度变化。最终能达到怎样的精度？

## 核心思想

ConvNeXt 的核心贡献是一张"现代化路线图"——把 ResNet-50 当起点，按顺序加入 7 类改造，每一步对应 ViT/Swin 中的某个设计。最终得到的网络**不含任何注意力机制**，仍然全部由卷积构成，但在 ImageNet 上超过同等规模的 Swin Transformer。

```mermaid
graph LR
    x["Input [B,3,224,224]"]:::input
    stem["Stem: Conv 4×4 / s=4 / 96 (patchify)"]:::compute
    s1["Stage 1: ConvNeXt Block × 3 / 96ch"]:::compute
    d1["Downsample: LN + Conv 2×2 / s=2"]:::compute
    s2["Stage 2: ConvNeXt Block × 3 / 192ch"]:::compute
    d2["Downsample: LN + Conv 2×2 / s=2"]:::compute
    s3["Stage 3: ConvNeXt Block × 9 / 384ch"]:::compute
    d3["Downsample: LN + Conv 2×2 / s=2"]:::compute
    s4["Stage 4: ConvNeXt Block × 3 / 768ch"]:::compute
    gap["Global Avg Pool + LN → [B,768]"]:::compute
    fc["FC 1000"]:::compute
    y["Softmax [B,1000]"]:::output

    x --> stem --> s1 --> d1 --> s2 --> d2 --> s3 --> d3 --> s4 --> gap --> fc --> y

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```
*图 1：ConvNeXt-T 主干，4 stage 的 ConvNeXt block 堆叠 (3,3,9,3)。*

整网骨架仍然是 4 stage 的"逐步降分辨率、逐步升通道"格式——这件事和 ResNet 完全一样。差别在 stem、block 内部、归一化 / 激活函数和训练 recipe。下面这张表是论文里那张著名的"现代化路径图"，每一行都对应 Swin 用到的某个设计：

| 改造 | 来自 | 精度变化（ResNet-50 → ） | 累积 Top-1 |
|---|---|---|---|
| 起点（ResNet-50 原版 recipe） | — | — | 76.1% |
| **新训练 recipe**（AdamW + 强增强 + 300 epoch） | ViT/Swin | +2.7 | 78.8% |
| **Stage 比例调整**（3,4,6,3 → 3,3,9,3） | Swin | +0.6 | 79.4% |
| **Patchify stem**（7×7 s=2 → 4×4 s=4 conv） | ViT | +0.1 | 79.5% |
| **ResNeXt 化 / depthwise conv** | ResNeXt | +1.0 | 80.5% |
| **Inverted bottleneck**（4× hidden） | MobileNet v2 | +0.1 | 80.6% |
| **大 kernel**（3×3 → 7×7 depthwise） | Swin window 注意力 | +0.7 | 80.6% |
| **更少 act/norm + GELU + LayerNorm** | Transformer | +0.7 | 81.5% |
| **独立 downsample 层**（LN + 2×2 s=2 conv） | Swin patch merging | +0.5 | 82.0% |

最终 **ConvNeXt-T 拿到 82.1% Top-1**，比同算力的 Swin-T（81.3%）高 0.8 点。把同一套配方放大到 T / S / B / L / XL 五档，**ConvNeXt-XL 在 ImageNet-22K 预训练 + 微调下达到 87.8% Top-1**，与 Swin-XL 持平或更优，同时 wall-clock 吞吐量更高（卷积比注意力更 GPU 友好）。

**改造细节中几条值得拎出来讲的：**

**Patchify stem**——原版 ResNet 的 stem 是 7×7 stride=2 conv + 3×3 maxpool，把分辨率一次性砍到 1/4，但卷积核之间有重叠。ViT 用的是不重叠的 16×16 patch（即 16×16 stride=16 conv），Swin 用 4×4 stride=4。ConvNeXt 取 4×4 stride=4——精度涨 0.1 点，本身意义不大，但**把后续所有 block 的输入对齐到"非重叠 patch"格式**，让 stage 之间的独立 downsample 层成为可能。

**Depthwise 7×7**——这一步是论文中关键的"算子级"改造。把 ResNet 原本的 3×3 卷积换成 **depthwise 7×7 卷积**——depthwise 控制 FLOPs（参数从 $C^2 k^2$ 降到 $C k^2$），7×7 把感受野扩到接近 Swin 的局部窗口（Swin-T window=7）。论文做过消融：3×3/5×5/7×7/9×9/11×11 五档，**精度在 7×7 饱和**，更大 kernel 不再带来提升。这一结果明确了"大 kernel 卷积可作为局部注意力的替代"这一观察的有效范围——不需要做到 attention 的全局，7×7 局部感受野配合层叠即可。

**LayerNorm 取代 BatchNorm**——另一个关键替换。ResNet 时代每个 conv 后接一个 BN，但 BN 依赖 batch 统计、对小 batch 不稳定、推理时需切换到 running mean 模式，工程上存在一些不便（详见 [foundations/04-normalization](../foundations/04-normalization/)）。Transformer 配套使用 LayerNorm——只对单样本的特征维做归一化，无 batch 依赖。ConvNeXt 把 BN 全替换为 LN（按 channel 维归一化的 2D 版本），精度提升 0.1 个点，但**训练在小 batch / 多机分布式下更稳定**，工程价值大于该 0.1 个点。

**减少 activation 和 norm 的数量**——ResNet 的每个 conv 后都跟 BN 和 ReLU，一个 Bottleneck block 内 3 个 conv 就有 3 组 BN + 3 个 ReLU。Transformer block 不是这样——一个 block 内只有一次 LN（在 attention 之前）和一次 LN（在 FFN 之前），激活函数 GELU 也只在 FFN 中间出现一次。ConvNeXt 照搬这个思路：**每个 block 内只保留 1 个 LayerNorm（在 depthwise conv 之后）和 1 个 GELU**（在两次 pointwise conv 之间）。少这几层归一化和激活，反而涨了 0.5 点——说明 ResNet 时代到处堆 BN+ReLU 的做法有冗余。

**ConvNeXt Block** 的最终形态：

```
x → DWConv 7×7 → LN → PWConv 1×1 (4× hidden) → GELU → PWConv 1×1 (↓) → DropPath → +x
      └────────────────────── shortcut ──────────────────────────────────────┘
```

注意这个顺序在结构上**和 Transformer 的 FFN block 几乎一一对应**——把 DWConv 看作"局部 token mixer"（相当于 self-attention 的位置），中间的两次 PWConv + GELU 就是标准的 MLP（相当于 FFN）。整个 ConvNeXt block 可以被理解为"用 7×7 depthwise conv 替代 self-attention 的 Transformer block"。

## 训练细节

ConvNeXt 的"训练 recipe 全套现代化"在 ResNet-50 那一行就给出了 +2.7 的精度收益——这是所有改造里**单步收益最大**的一项，比任何结构改造都重要。这件事必须单独列出来。

| 维度 | ResNet-50 原版（2015） | ConvNeXt（2022） |
|---|---|---|
| 优化器 | SGD + Momentum 0.9 | **AdamW** (β₁=0.9, β₂=0.999) |
| 学习率 | 0.1，阶梯 decay | 4×10⁻³，**cosine decay** |
| 学习率 warmup | 无 | **20 epoch linear warmup** |
| 权重衰减 | 1×10⁻⁴ | 0.05 |
| Batch size | 256 | **4096** |
| Epochs | 90–120 | **300** |
| 激活函数 | ReLU | **GELU** |
| 归一化 | BatchNorm | **LayerNorm** |
| 标签平滑 | 无 | Label Smoothing 0.1 |
| Mixup | 无 | **α=0.8** |
| CutMix | 无 | **α=1.0** |
| RandAugment | 无 | **(9, 0.5)** |
| RandomErasing | 无 | **p=0.25** |
| Stochastic Depth | 无 | **0.1 (T) → 0.5 (XL)** |
| EMA | 无 | **decay=0.9999** |

**AdamW 取代 SGD** 是这套 recipe 里另一个有结构性影响的选择。SGD 在 CNN 时代是默认优化器，但它对学习率 schedule 和初始化非常敏感；AdamW 把 weight decay 从梯度更新里解耦出来（详见 [foundations/03-optimizers](../foundations/03-optimizers/)），在 Transformer 时代成了大模型训练的默认配置。ConvNeXt 把这条经验搬回 CNN——在 batch 4096、cosine schedule、20 epoch warmup 的现代训练设定下，AdamW 比 SGD 稳得多，最终精度也更高。

**强数据增强组合** 是另一项关键。Mixup + CutMix + RandAugment + RandomErasing + Label Smoothing + Stochastic Depth——这套组合由 DeiT（2021）为让 ViT 在 ImageNet-1K 上不依赖 JFT-300M 预训练也能达到高精度而调出。把这套组合应用到 ResNet-50 训练 300 epoch，仅此一项即可将精度从 76.1% 提升到 78.8%。这一结果在 2022 年之前已有研究观察到（如 *ResNet Strikes Back*, Wightman 2021），ConvNeXt 将其正式纳入 baseline。

**训练资源**：ConvNeXt-T 在 8 块 A100 上训 300 epoch 约 2 天，ConvNeXt-XL 在 32 块 A100 上训 ImageNet-22K 预训练约 1 周 + ImageNet-1K 微调 1 天。

**ImageNet Top-1 对比（224×224 输入，ImageNet-1K only）：**

| 模型 | 参数量 | FLOPs | Top-1 |
|---|---|---|---|
| ResNet-50（2015 recipe） | 25.6M | 4.1B | 76.1% |
| ResNet-50（现代 recipe） | 25.6M | 4.1B | 78.8% |
| EfficientNet-B4 | 19M | 4.2B | 82.9% |
| Swin-T | 28M | 4.5B | 81.3% |
| **ConvNeXt-T** | **29M** | **4.5B** | **82.1%** |
| Swin-B | 88M | 15.4B | 83.5% |
| **ConvNeXt-B** | **89M** | **15.4B** | **83.8%** |

**ImageNet-22K 预训练 + 1K 微调（384×384 输入）：**

| 模型 | 参数量 | Top-1 |
|---|---|---|
| Swin-L | 197M | 87.3% |
| **ConvNeXt-L** | **198M** | **87.5%** |
| Swin-XL（CLIP/SwinV2） | 350M | 87.6% |
| **ConvNeXt-XL** | **350M** | **87.8%** |

87.8% 这一数值是 2022 年 CNN 在 ImageNet 上对齐并略超 Swin 的代表性结果——同参数下 ConvNeXt 与 Swin 持平或略高，同时**推理 throughput 比 Swin 高约 20%**（卷积比 window attention 更 GPU 友好）。

## 关键代码

下面这段实现 ConvNeXt block 的核心：depthwise 7×7 → LN → pointwise (4× hidden) → GELU → pointwise (↓) → DropPath → + shortcut。注意一个细节：LN 在 channel-last 格式上算（即 [B, H, W, C]），所以中间有 permute——这是 ConvNeXt 实现里最容易踩坑的一点。

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt 的基本砖：DWConv 7×7 → LN → PWConv↑ → GELU → PWConv↓ → DropPath + shortcut。"""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        # Depthwise 7×7：局部 token mixer，替代 self-attention 的位置
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm 作用在 channel 维（channel-last 格式）
        self.norm   = nn.LayerNorm(dim, eps=1e-6)
        # 两次 1×1 pointwise：先升 4× 再降回，对应 Transformer 的 FFN
        self.pwconv1 = nn.Linear(dim, 4 * dim)         # 用 Linear 实现 1×1 pwconv（更快）
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Layer Scale：每个通道一个可学习的缩放系数，初值极小（1e-6）
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        # Stochastic Depth：按概率随机丢整个 block
        self.drop_path = nn.Identity() if drop_path == 0.0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                                   # [B, C, H, W]
        x = self.dwconv(x)                             # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)                      # → [B, H, W, C] for LN/Linear
        x = self.norm(x)
        x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        x = self.gamma * x                             # Layer Scale
        x = x.permute(0, 3, 1, 2)                      # → [B, C, H, W]
        return identity + self.drop_path(x)            # 残差连接，沿用 ResNet
```

整个 ConvNeXt-T 就是 stem (4×4 s=4) + 4 stage 的 ConvNeXt Block 堆叠 (3, 3, 9, 3)，stage 之间用 "LN + 2×2 s=2 conv" 做独立 downsample。`identity + drop_path(x)` 这一行就是 ConvNeXt 继承自 ResNet 的最重要遗产——shortcut connection 七年后依然没变。

## 影响 / 后续

ConvNeXt 是 CNN 家族这条主线上**近期具有重要影响的工作之一**。其结果——ImageNet-22K 预训练下 **87.8% Top-1**，吞吐量高于 Swin——对"CNN 是否被 Transformer 替代"这一 2020–2021 年讨论较多的问题给出了一定的回答：在视觉这一具体任务上，**架构设计的现代化选择对结果的影响大于卷积 vs 注意力的算子之争**，CNN 的局部归纳偏置 + Transformer 的现代训练方法可以视为互补而非互斥的组合。

但 ConvNeXt 的优势是局部的。**2022 年之后视觉主线整体转向 ViT 路线**——原因不在 ImageNet 单任务的精度，而在更下游的方向（多模态、视觉-语言对齐、大规模自监督预训练、SAM 这类通用视觉基础模型、CLIP 这类跨模态对齐）多数沿 Transformer 路线展开。ConvNeXt 2023 年发布 V2 版本，加入 GRN（Global Response Normalization）和 MAE 风格的自监督预训练，使 CNN 在自监督方向上跟上 ViT 的水平——但社区关注度已转向多模态大模型，纯视觉 backbone 的研究关注度有所下降。

回望整条 CNN 演化路径：[AlexNet](02-alexnet.md) 让端到端学到的特征超过手工特征 → [VGG](03-vgg.md) / [Inception](04-inception.md) 把深度推到 20 层 → [ResNet](05-resnet.md) 用 shortcut 实现 152 层稳定训练 → [DenseNet](06-densenet.md) / [EfficientNet](07-efficientnet.md) 在参数效率方向上持续优化 → ConvNeXt 用现代训练方法说明 CNN 在视觉任务上仍具竞争力。这条十年的路径在 ConvNeXt 处形成了一个相对完整的阶段。视觉主线之后的发展请参考 ViT 章节。

→ [../08-vit/](../08-vit/) · 视觉主线已移交，多模态 / 大模型大多从 ViT 路线展开
→ [../foundations/02-activations/](../foundations/02-activations/) · GELU 是 Transformer/ConvNeXt 的标配
→ [../foundations/04-normalization/](../foundations/04-normalization/) · LayerNorm 取代 BatchNorm 是关键改造之一
→ [../foundations/03-optimizers/](../foundations/03-optimizers/) · AdamW 取代 SGD 在大模型时代成为新标配
