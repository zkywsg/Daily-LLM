---
name: "EfficientNet"
year: 2019
family: "01-cnn"
order: 7
paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
authors: ["Mingxing Tan", "Quoc V. Le"]
key_idea: "用复合缩放系数把 depth/width/resolution 三轴联合缩放公式化，得到帕累托最优的 B0–B7 模型族"
---

# EfficientNet (2019)

## 前作进展

[ResNet](05-resnet.md) 之后，CNN 的"做大"这件事被拆成了三个旋钮：**加深**（更多层）、**加宽**（更多通道）、**加分辨率**（更大输入图）。每条路都各自被验证过——ResNet 把深度从 22 推到 152，WideResNet 把宽度乘到 10 倍，多尺度训练把输入从 224 拉到 600。三条线各自能换来精度，但三者之间的关系没人系统问过。

实际操作中，常见的做法是"一次只调一个旋钮，凭经验"。需要更高精度时把 ResNet-50 改为 ResNet-200；需要参数更少时走 [Inception](04-inception.md) 路线手工配通道；要在限定 FLOPs 预算下提升精度时，MobileNet 减小宽度、降低输入分辨率，每组超参依赖手工调试。**尚未有系统性的研究回答一个基础问题**：固定 FLOPs 预算下，深度、宽度、分辨率三者应如何联合分配？

另一个问题是，单独增加任一轴都会较快出现边际收益递减——只加深时 200 层后精度饱和；只加宽时参数显著上升但精度提升有限；只加分辨率时 FLOPs 按平方增长而精度仅线性提升几个点。Tan 和 Le 在做 NAS 工作（MnasNet）时观察到：**手工调试的模型族中，规模较大的模型倾向于同时加深、加宽、加分辨率，且三者的比例相对稳定**。这一经验性的观察推动了 EfficientNet——能否将"三轴联合放大"形式化为一个数学公式？

## 核心思想

EfficientNet 把"放大网络"重新表述成一个带约束的优化问题。设有一个种子模型，它的深度、宽度、分辨率为基准 1。引入单一**复合缩放系数 φ**（compound coefficient），三轴按下式联合缩放：

$$
\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi
$$

其中 $\alpha, \beta, \gamma \geq 1$ 是三个常数，满足约束：

$$
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

这条约束有非常具体的物理含义：卷积的 FLOPs 与 depth 成正比、与 width² 成正比（每层输入通道 × 输出通道）、与 resolution²（H×W）成正比。所以 $\alpha \cdot \beta^2 \cdot \gamma^2$ 恰好是 FLOPs 的总缩放系数。把它钉在 2，意味着 **φ 每加 1，整网 FLOPs 翻倍**。φ=0 是基准模型 B0，φ=1, 2, …, 7 依次得到 B1–B7。

α, β, γ 的具体值通过在 B0 上做一次小规模 grid search 得到：**α=1.2, β=1.1, γ=1.15**——这意味着 FLOPs 翻倍时，深度涨 20%、宽度涨 10%、分辨率涨 15%。比例并不对等：宽度比深度便宜（宽度对 FLOPs 是平方贡献，所以同样翻倍预算下分给宽度的 exponent 必须小），分辨率也是平方贡献。这组数字一旦定下来，B1–B7 全部按 φ 推出来，**不再有任何额外的手工调超参**。

种子模型 EfficientNet-B0 自身是用 NAS 在"FLOPs ≈ 400M、精度最优"的目标下搜出来的——和 MnasNet 同一套搜索空间。结构上 B0 由一个 stem（3×3 conv）+ 7 个 stage 的 **MBConv**（Mobile Inverted Bottleneck Conv，源自 MobileNet v2）堆叠 + head（1×1 conv + GAP + FC）构成。MBConv 内部用 **inverted bottleneck**（1×1 升维 → depthwise 3×3 → 1×1 降维）+ **Squeeze-Excitation**（源自 SE-Net）+ Swish 激活，这些都是横切组件——本节不展开，留给 [foundations/02-activations](../foundations/02-activations/) 和后续 MobileNet/SENet 专题。

```mermaid
graph LR
    x["Input [B,3,224,224]"]:::input
    stem["Stem: Conv 3×3 / s=2 / 32"]:::compute
    s1["Stage 1: MBConv1 / k=3×3 / 16ch × 1"]:::compute
    s2["Stage 2: MBConv6 / k=3×3 / 24ch × 2 / s=2"]:::compute
    s3["Stage 3: MBConv6 / k=5×5 / 40ch × 2 / s=2"]:::compute
    s4["Stage 4: MBConv6 / k=3×3 / 80ch × 3 / s=2"]:::compute
    s5["Stage 5: MBConv6 / k=5×5 / 112ch × 3"]:::compute
    s6["Stage 6: MBConv6 / k=5×5 / 192ch × 4 / s=2"]:::compute
    s7["Stage 7: MBConv6 / k=3×3 / 320ch × 1"]:::compute
    head["Head: Conv 1×1 / 1280 + GAP"]:::compute
    fc["FC 1000"]:::compute
    y["Softmax [B,1000]"]:::output

    x --> stem --> s1 --> s2 --> s3 --> s4 --> s5 --> s6 --> s7 --> head --> fc --> y

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```
*图 1：EfficientNet-B0 主干，7 个 stage 的 MBConv 堆叠。MBConv1/6 中数字为 expansion ratio，每 stage 标 kernel / 输出通道 / block 数 / stride。*

**三轴等比 vs 单轴缩放**——论文中的关键消融是在"相同 FLOPs 预算"下对比四种缩放策略：只加深、只加宽、只加分辨率、三轴等比。**同 FLOPs 下三轴联合缩放比任一单轴策略高 0.5–2.5 个 Top-1 百分点**。这条曲线为"复合缩放"提供了实证依据——φ 这一公式是基于实测的帕累托线，而非经验性选择。

最终得到一个完整的模型族 B0–B7。B0 用 5.3M 参数达到 77.1% Top-1，B7 用 66M 参数达到 **84.3% Top-1（2019 ImageNet SOTA）**。同精度下 EfficientNet-B7 比 GPipe 小 8.4×、比 ResNeXt-101 小数倍——"参数效率"这一由 Inception 开启的路线，在 EfficientNet 上进一步推进。

## 工程陷阱

**α, β, γ 是在 B0 上搜得的，换基础架构未必最优**。这组常数 (1.2, 1.1, 1.15) 是在 EfficientNet-B0（MBConv + SE）这一特定种子模型上 grid search 得到的。直接搬到 ResNet、ConvNeXt 或其他 backbone 上做 "compound scaling"，比例不一定最优。Tan 和 Le 在 **EfficientNet-V2（2021）** 中发现：**width 应当更激进、resolution 增长应放缓**——大分辨率训练显存开销大、训练速度下降明显。V2 重新搜索了范围，最优比例与 V1 不同。**复合缩放是一个框架，但具体 α, β, γ 是 backbone 相关的**，迁移到其他架构时需要重搜。

**Stochastic Depth 是大模型保精度的关键超参，早期实现常忽略**。EfficientNet-B0 训练时 stochastic depth 的丢弃率为 0，看似无关紧要——但 B4 之后丢弃率线性增长至 0.2（B7），论文消融实验显示 **B7 关闭 stochastic depth 时 Top-1 下降 0.7–1.0 个百分点**。2019 年多数第三方复现（包括 PyTorch torchvision 早期版本）默认忽略这一超参，结果 B5–B7 比论文低约 1 个百分点，社区耗时数月才定位到这一差异。**模型放大时正则强度也需要同步增加**，stochastic depth 是 EfficientNet 大模型族的重要组件。

**Depthwise conv 的 FLOPs 便宜不等于 wall-clock 速度快**。MBConv 大量使用 depthwise 3×3/5×5，理论 FLOPs 远低于普通卷积——但在 2019 年的 GPU 上（V100、T4），depthwise conv 的 cuDNN 实现优化程度低于标准卷积，实测 wall-clock 时间**只快 1.5–2×**，而非 FLOPs 比例所暗示的 8–9×。这导致一个反直觉现象：**EfficientNet-B0 的 FLOPs 是 ResNet-50 的 1/10，但实际推理速度只快 2–3 倍**。同精度下 EfficientNet-B3（FLOPs 与 ResNet-50 相当）的延迟反而比 ResNet-50 高。**部署选型时不能仅依据 FLOPs，需在目标硬件上实测 latency**。这也是 EfficientNet-V2 把 stage 1–3 换回普通卷积（Fused-MBConv）的原因——小分辨率高通道阶段，depthwise 的硬件不友好性较明显，换回普通 conv 整体更快。

## 训练细节

| 维度 | 值 |
|---|---|
| 优化器 | **RMSProp**（不是 SGD），decay=0.9，momentum=0.9 |
| 初始学习率 | **0.256**（batch size 4096 下），decay 0.97 每 2.4 epochs |
| 权重衰减 | 1×10⁻⁵ |
| Dropout（FC 前） | B0=0.2，B1=0.2，…，B7=0.5（随模型变大线性涨） |
| Stochastic Depth 丢弃率 | B0=0.0，B4=0.2，B7=0.2（按 block 深度线性 schedule） |
| Batch size | 4096（TPU pod 跨核心累积） |
| Epochs | 350 |
| 激活函数 | **Swish/SiLU**（$x \cdot \sigma(x)$），全网替换 ReLU |
| 归一化 | BatchNorm，momentum=0.99 |

**为什么 RMSProp、为什么 lr=0.256**——这两条都不是 ImageNet 训练的"主流"配置（主流是 SGD + 0.1）。原因是 EfficientNet 沿用了 MnasNet 的训练 recipe，MnasNet 当年是为了在 TPU 上做 NAS 搜索而调的，RMSProp 在 TPU pod 上对各种结构的稳健性比 SGD 好。0.256 = 0.016 × 16（batch 4096 vs 256 的 lr 线性 scaling）。复现时如果换到 GPU + 小 batch，**必须按比例 rescale 学习率**，否则不收敛。

**数据增强**：

- **AutoAugment**（ImageNet policy）—— 用强化学习搜出来的 25 条增强子策略，每张训练图随机走一条
- **RandomErasing**（部分模型用）
- **MixUp**（B5+ 才用，alpha=0.2）
- **测试时增强**：B7 用更高分辨率（600×600）训练 + 测试时单 crop，不做多 crop ensemble

**训练资源**：B0 在 TPUv3 pod 上 ~7 小时，B7 ~36 小时；论文实验全部在 Google 内部 TPU pod 上完成。第三方在 GPU 上复现 B7 通常需要 8×V100 训练约一周。

**ImageNet 错误率（Top-1）：**

| 模型 | 参数量 | FLOPs | Top-1 |
|---|---|---|---|
| ResNet-50 | 25.6M | 4.1B | 76.0% |
| ResNet-152 | 60.2M | 11.5B | 78.3% |
| ResNeXt-101 (64×4d) | 84M | 31.5B | 80.9% |
| GPipe | 557M | — | 84.3% |
| **EfficientNet-B0** | **5.3M** | **0.39B** | **77.1%** |
| EfficientNet-B3 | 12M | 1.8B | 81.6% |
| EfficientNet-B5 | 30M | 9.9B | 83.6% |
| **EfficientNet-B7** | **66M** | **37B** | **84.3%** |

B7 用 1/8.4 的参数追平了 GPipe 的 SOTA——这一结果定义了 2019 年新的帕累托线，2022 年被 ConvNeXt 超越。

## 关键代码

下面这段实现 MBConv 的核心结构：**1×1 升维（expand）→ depthwise 3×3 → SE → 1×1 降维（project）+ shortcut**。BN/Swish 按 EfficientNet 默认顺序：

```python
import torch
import torch.nn as nn

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck + Squeeze-Excitation，EfficientNet 的核心砖。"""

    def __init__(self, in_c: int, out_c: int, expand: int = 6,
                 kernel: int = 3, stride: int = 1, se_ratio: float = 0.25):
        super().__init__()
        mid_c = in_c * expand                        # inverted bottleneck：先升维
        self.use_shortcut = (stride == 1 and in_c == out_c)

        # 1×1 升维（expand=1 时跳过，B0 第一个 stage 就是这种情况）
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c), nn.SiLU(inplace=True),
        ) if expand != 1 else nn.Identity()

        # Depthwise k×k（groups=mid_c 即深度可分离卷积）
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel, stride, kernel // 2,
                      groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c), nn.SiLU(inplace=True),
        )

        # SE：global pool → 压到 1/4 → 升回 → sigmoid 门控
        se_c = max(1, int(in_c * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_c, se_c, 1), nn.SiLU(inplace=True),
            nn.Conv2d(se_c, mid_c, 1), nn.Sigmoid(),
        )

        # 1×1 降维到 out_c（注意：project 之后无激活——线性瓶颈）
        self.project = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dwconv(out)
        out = out * self.se(out)                     # SE 门控
        out = self.project(out)
        return out + x if self.use_shortcut else out  # 仅同 shape 时加 shortcut
```

整个 EfficientNet-B0 就是把这种 MBConv 按 (16, 24, 40, 80, 112, 192, 320) 通道、(1, 2, 2, 3, 3, 4, 1) block 数、(3, 3, 5, 3, 5, 5, 3) kernel 配置堆 7 个 stage。B1–B7 把每个 stage 的 block 数按 α^φ 涨、把每层通道按 β^φ 涨、把输入分辨率按 γ^φ 涨，**结构骨架完全不变**——这就是 compound scaling 在代码层面的全部含义。

## 影响 / 后续

EfficientNet 在 2019–2021 年间在"参数效率"坐标轴上占据主导地位——detection、segmentation、医学影像、移动端部署等需要在精度与参数间寻找帕累托点的场景，EfficientNet 常作为默认基线。它把 "compound scaling" 留作一个**架构无关的设计范式**：后来 RegNet、EfficientNet-V2、NFNet 都借用了"用单参数控制模型族放大"的思路，只是重新搜索 α/β/γ 或将其改为可学习。

EfficientNet 自身的局限也较快显现。**Wall-clock 速度与 FLOPs 不一致**是工业界长期关注的问题；2020 年后 ConvNeXt / ViT 兴起，CNN 在 SOTA 上的优势 2022 年被 ConvNeXt 通过"训练 recipe 现代化"的方式追平——同样的 ResNet 骨架，只要把 AdamW + Swish + LayerScale + 大模型训练技巧引入，就能在精度上超过 EfficientNet-B7 的帕累托线。这也提示：**架构本身的设计空间已接近边际收益，未来几年 CNN 的提升更多来自训练侧而非结构侧**。

→ [08-convnext.md](08-convnext.md) · 把 ViT 的训练方法反哺到 CNN，超越 EfficientNet 帕累托线
→ [../foundations/04-normalization/](../foundations/04-normalization/) · MBConv 里 BN 和 inverted bottleneck 的搭配
→ [../foundations/02-activations/](../foundations/02-activations/) · MBConv 用 Swish/SiLU 替代 ReLU，激活函数演化中的一站
