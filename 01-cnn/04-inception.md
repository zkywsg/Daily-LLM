---
name: "GoogLeNet (Inception v1)"
year: 2014
family: "01-cnn"
order: 4
paper: "Going Deeper with Convolutions"
authors: ["Christian Szegedy", "Wei Liu", "Yangqing Jia", "Pierre Sermanet", "Scott Reed", "Dragomir Anguelov", "Dumitru Erhan", "Vincent Vanhoucke", "Andrew Rabinovich"]
key_idea: "用 1×1 卷积降维 + 多尺度并行的 Inception 模块，把参数量压到 VGG 的 1/12 同时拿下 ImageNet 冠军"
---

# GoogLeNet / Inception v1 (2014)

## 前作进展

2014 年时，"更深更好"已被 [VGG](03-vgg.md) 实证——所有卷积统一为 3×3、堆叠至 16/19 层，Top-5 错误率降至 7.3%，接近人类水平。但 VGG 也暴露出一个明显的参数分布问题：**138M 参数中 102M 集中在 fc6 一层**，网络名义上是"深 CNN"，但参数主要分布在 FC 层。

继续加深的方案面临较大开销：VGG-16 单模型推理约 15 GFLOPs、参数 138M，难以部署到移动端或大规模在线服务。若按 VGG 思路继续做到 30 层、50 层，参数和算力会显著增长；而模型规模增大在 ImageNet 上的边际收益也在下降——VGG-19 相比 VGG-16 的提升已很有限。

当时的主要思路是：要进一步提升精度，要么继续加深（参数代价大），要么寻找一种**在不显著增加参数的前提下提升网络表达能力**的新结构。Inception 走的是后一条路径。

## 核心思想

GoogLeNet 没有继续往"更深更窄"那条路上走，而是回到一个更基本的问题：**视觉模式天然没有单一尺度**。一张图里既有覆盖整张脸的轮廓、也有几个像素的眼角细节；一个分类器要应对所有这些，理论上应该在每一层同时观察多个尺度。

VGG 的做法是"统一使用一个尺度（3×3）再堆深"，依靠堆叠扩大感受野。Inception 的做法是：**在同一层并行使用多种尺度的卷积，再把结果拼接**——1×1 看通道关系、3×3 看小邻域、5×5 看更大邻域、3×3 MaxPool 提供位置不变性。最后在通道维度 concat，由下一层自适应学习如何加权这些通道。

```mermaid
graph TD
    x["Input [B,3,224,224]"]:::input
    stem["Stem: Conv 7×7/s=2/64 + MaxPool 3×3/s=2 + Conv 3×3/192 + MaxPool 3×3/s=2"]:::compute
    i3a["Inception 3a/3b (× 2)"]:::compute
    p3["MaxPool 3×3 / s=2"]:::compute
    i4["Inception 4a–4e (× 5)"]:::compute
    p4["MaxPool 3×3 / s=2"]:::compute
    i5["Inception 5a/5b (× 2)"]:::compute
    gap["Global Avg Pool 7×7 → [B,1024]"]:::compute
    drop["Dropout p=0.4"]:::compute
    fc["FC 1000"]:::compute
    y["Softmax [B,1000]"]:::output

    aux1["Aux Head @ 4a"]:::compute
    aux2["Aux Head @ 4d"]:::compute

    x --> stem --> i3a --> p3 --> i4 --> p4 --> i5 --> gap --> drop --> fc --> y
    i4 -.-> aux1
    i4 -.-> aux2

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```
*图 1：GoogLeNet 整体结构——stem + 9 个 Inception block + GAP + 单层 FC。两个辅助分类器只在训练时挂在 4a/4d 后面。*

这条结构里，GoogLeNet 含 22 层有参层（不算池化），9 个 Inception block 占绝大部分计算。整网参数约 **5M**——比 VGG-16 (138M) 少约 28 倍，比 AlexNet (60M) 少约 12 倍。ImageNet Top-5 错误率为 **6.67%**（VGG 7.3%），获得 ImageNet 2014 冠军。

但简单的多分支并行存在一个严重问题——**通道数会随分支累积**。假设输入 256 通道，每分支各输出 128 通道，concat 后变成 512 通道，下一层的 5×5 卷积就需要在 512 通道上计算，参数量为 $5 \times 5 \times 512 \times 128 = 1.6\text{M}$ 一层。堆叠几个 block 后参数量就会回到 VGG 的量级。

Inception 的核心设计是**在每个 3×3 / 5×5 之前先用 1×1 卷积做"瓶颈降维"**——把输入通道从 256 降到 64 再做 5×5，参数量降至 $5 \times 5 \times 64 \times 128 = 0.2\text{M}$，**约为原来的 1/8**。1×1 卷积只对通道维做线性组合（不改变空间维），计算便宜且能学到通道压缩，是 Inception 设计的核心机制。

```mermaid
graph TD
    in["Input [B,C_in,H,W]"]:::input
    b1["1×1 Conv / C₁"]:::compute
    b2a["1×1 Conv / C₂ʳ (降维)"]:::compute
    b2b["3×3 Conv / C₂"]:::compute
    b3a["1×1 Conv / C₃ʳ (降维)"]:::compute
    b3b["5×5 Conv / C₃"]:::compute
    b4a["3×3 MaxPool / s=1"]:::compute
    b4b["1×1 Conv / C₄ (降维)"]:::compute
    cat["Concat (通道维)"]:::compute
    out["Output [B,C₁+C₂+C₃+C₄,H,W]"]:::output

    in --> b1 --> cat
    in --> b2a --> b2b --> cat
    in --> b3a --> b3b --> cat
    in --> b4a --> b4b --> cat
    cat --> out

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```
*图 2：Inception block 内部——4 分支并行，3×3/5×5 之前先用 1×1 瓶颈降维，最后在通道维 concat。*

形式上，Inception block 的输出可以写成 4 路分支在通道维的拼接：

$$
y = \text{Concat}\Big(\, f_{1\times 1}(x),\; f_{3\times 3}(g^{(2)}_{1\times 1}(x)),\; f_{5\times 5}(g^{(3)}_{1\times 1}(x)),\; g^{(4)}_{1\times 1}(\text{MaxPool}(x)) \,\Big)
$$

每条分支的 $g_{1\times 1}$ 把输入通道压低后再走更贵的 3×3 / 5×5。这种"先压再算"的结构后来在 ResNet 的 bottleneck block、MobileNet 的 inverted residual 里被反复借用。

**用 GAP 取代大 FC** —— GoogLeNet 的另一个关键设计是：去掉 VGG/AlexNet 那两层 4096 维的 FC。最后一个 Inception block 输出 $7 \times 7 \times 1024$，直接做 Global Average Pooling 把每个通道平均为 1 个数，得到 1024 维向量，再接一层 FC 到 1000 类。这一改动把参数从 VGG 的 102M 降至约 1M，**是 5M 整网参数预算的主要来源**。GAP 这个技巧（连同 1×1 卷积）来自 Lin 等人 2013 年的 "Network in Network"，Inception 是它在大规模视觉模型上的首次应用。

## 工程陷阱

**1×1 降维必须在 3×3/5×5 之前，顺序反了就失去意义**。1×1 的目的是把输入通道砍下来，让后面贵的 3×3/5×5 在低维上算。如果顺序写反（先 3×3 再 1×1），3×3 已经在高维输入上算完了，1×1 只能压缩输出通道、对 3×3 自身的计算量毫无帮助。同理 pool 分支的 1×1 放在 MaxPool **之后**——因为 MaxPool 不改变通道数，只在 pool 之后的拼接前压一下。

**辅助分类器是 BN 出现前的临时补丁**。GoogLeNet 在 inception-4a 和 inception-4d 后面各挂一个辅助 head（小 FC 接 softmax），训练时它们各自算一遍交叉熵 loss，按 0.3 的权重加进总 loss：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + 0.3 \, \mathcal{L}_{\text{aux1}} + 0.3 \, \mathcal{L}_{\text{aux2}}
$$

其作用主要是**缓解梯度消失**——22 层网络的最底层在 BN 出现前梯度信号衰减较大，从中间层直接接 softmax 相当于给底部"增加两条短回路"让梯度能够回传。**推理时这两个 head 不参与最终预测，直接舍弃**。这种"辅助监督"方案在 BN（2015）和残差连接（2015）出现后逐渐不再使用——[ResNet](05-resnet.md) 之后辅助分类器在主流工作中较少出现。

**分支选择数和通道分配依赖手工调试**。原论文中每个 Inception block 的 4 分支输出通道（$C_1, C_2, C_3, C_4$）和 2 个降维通道（$C_2^r, C_3^r$）均为查表的固定值——3a 是 (64, 128, 32, 32) + (96, 16)，3b 是 (128, 192, 96, 64) + (128, 32)，每个 block 一组不同的数值。这种"手工调超参"的特点较明显，复现实验时改任一数值都可能影响结果。该问题直到 [EfficientNet](07-efficientnet.md) 用 NAS + compound scaling 自动搜索结构后得到改善；Inception 自身在 v3 用 factorized convolution（7×7 → 1×7 + 7×1）也部分缓解了这个问题。

**concat 的通道顺序不可换**。4 分支在通道维拼接，下一层 Inception 的 1×1 降维会重新学习权重，原则上"哪个分支放第 0 通道"无所谓——但加载预训练权重时，分支顺序必须与原实现一致，否则权重对应错位。这是迁移学习时的常见注意事项。

## 训练细节

| 维度 | 值 |
|---|---|
| 优化器 | SGD + Momentum |
| 学习率 | 0.045（论文报告值），每 8 epoch 衰减 4% |
| 动量 | 0.9 |
| 辅助 head 权重 | 0.3 ×（aux1）+ 0.3 ×（aux2），仅训练时启用 |
| Dropout | p=0.4，仅 GAP 之后的 FC 前 |
| 权重初始化 | 不再用"先训浅版当种子"的接力——网络相对短且参数少 |
| 推理时辅助 head | 丢弃 |

**数据增强**：

- **多尺度裁剪**：从 8% 到 100% 的图像面积随机采样，宽高比在 [3/4, 4/3] 之间随机
- **光度扰动**：参照 Andrew Howard 2013 的方案做亮度/对比度/颜色扰动
- **测试时多裁剪**：取 144 个 crop（4 尺度 × 3 区块 × 2 翻转 × 6 crop）平均 softmax

**训练资源**：原论文用一组分布式 CPU 集群训练 GoogLeNet 单模型（DistBelief 框架），具体训练时长论文没给死。这个细节也反映了当年的工程现实——Google 内部当时还没大规模铺 GPU 训练框架。

**ImageNet 错误率（Top-5）：**

| 年份 | 方法 | 参数量 | Top-5 错误率 |
|---|---|---|---|
| 2012 | AlexNet | 60M | 15.3% |
| 2013 | ZFNet | 60M | 14.8% |
| 2014 | VGG-16 | 138M | 7.3% |
| 2014 | **GoogLeNet single** | **5M** | **7.89%** |
| 2014 | **GoogLeNet 7-model ensemble** | — | **6.67%** |

6.67% 这个冠军成绩的意义不仅在于优于 VGG，更在于**仅用 VGG 1/28 的参数量**——这是"参数效率"作为独立优化目标首次在 ImageNet 上取得领先位置。Inception 后续 v2（BN-Inception, 2015）、v3（factorized conv, 2015）、v4 / Inception-ResNet（2016）持续演化，"多分支 + 1×1 降维"的骨架被后续工作沿用。

## 关键代码

下面这段用 PyTorch 写一个标准 Inception block——4 分支、3×3/5×5 前用 1×1 降维、最后 concat：

```python
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    """单个 Inception v1 block：4 分支并行 + 1×1 瓶颈降维 + 通道维 concat。"""

    def __init__(self, in_c: int,
                 c1: int, c2r: int, c2: int, c3r: int, c3: int, c4: int):
        super().__init__()
        # 分支 1：纯 1×1
        self.b1 = nn.Sequential(nn.Conv2d(in_c, c1, 1), nn.ReLU(inplace=True))
        # 分支 2：1×1 降维 → 3×3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, c2r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c2r, c2, 3, padding=1), nn.ReLU(inplace=True),
        )
        # 分支 3：1×1 降维 → 5×5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, c3r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, 5, padding=2), nn.ReLU(inplace=True),
        )
        # 分支 4：3×3 MaxPool → 1×1 降维
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, c4, 1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 4 分支并行，输出在通道维拼接 → [B, c1+c2+c3+c4, H, W]
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

# Inception-3a 的官方配置：in=192, (c1, c2r, c2, c3r, c3, c4) = (64, 96, 128, 16, 32, 32)
# 输出通道数 = 64 + 128 + 32 + 32 = 256
block_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
```

整个 GoogLeNet 就是把这种 block 按论文表里给的通道配置堆 9 次，再加 stem、GAP、单层 FC——5M 参数的来源一目了然。

## 影响 / 后续

GoogLeNet 的结果——Top-5 错误率 **6.67%**、参数仅 **5M**——确立了"参数效率"作为独立优化目标的地位。此后视觉论文普遍同时报告精度和参数 / FLOPs，这种**双轴评估**的习惯由 Inception 推动形成。

Inception 的影响在多个方向延续。其"多分支 + 通道分配"骨架开拓了一条新路径——表达能力可通过并行结构而非单纯深度获得——这条路在 Inception v2/v3/v4 内部继续演化，也被 ResNeXt（多分支 + group conv）、Xception（深度可分离卷积，即 1×1 + depthwise 的极端形式）沿用。GAP + 1×1 卷积成为后续视觉模型的常见配置。辅助分类器在 [ResNet](05-resnet.md) 和 BatchNorm 出现后逐渐不再使用——可视为 2014 年深层网络训练不稳定时期的过渡方案。

→ [05-resnet.md](05-resnet.md) · 残差连接让深网络稳定训练，辅助分类器在后续工作中较少使用
→ [07-efficientnet.md](07-efficientnet.md) · 把 Inception "分支选择 / 通道分配"的手工调超参换成 NAS + compound scaling 自动搜索
→ [../foundations/04-normalization/](../foundations/04-normalization/) · BN-Inception（v2, 2015）是 BatchNorm 第一次在大规模视觉模型上落地
→ [03-vgg.md](03-vgg.md) · 同年的 VGG 走的是"深度堆叠"路线，Inception 证明了"参数效率"是另一条可走的轴
