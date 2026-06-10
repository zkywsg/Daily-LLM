---
name: "DenseNet"
year: 2017
family: "01-cnn"
order: 6
paper: "Densely Connected Convolutional Networks"
authors: ["Gao Huang", "Zhuang Liu", "Laurens van der Maaten", "Kilian Q. Weinberger"]
key_idea: "每层都直接接收前面所有层的输出（concat 而非加法），把特征复用推到极致"
---

# DenseNet (2017)

## 前作进展

[ResNet](05-resnet.md) 用 $y = F(x) + x$ 这条 shortcut 有效解决了 152 层的训练问题，残差连接成为 2015–2016 年深层网络的默认配置。但 ResNet 留下了一个值得关注的问题——**加法（add）操作在信息保留上有损失**。

`F(x) + x` 把不同层学到的特征**叠加到同一个张量上**。对优化来说这是优点，shortcut 给梯度一条干净的回流路径；但从信息保留的角度看，加法是一种**混叠**——浅层的边缘特征与深层的语义特征被加成一个张量，下游层无法区分"这个数值由哪一层贡献"。如果某一层的特征不需要被后续修改、只需要被原样使用，加法无法满足这种需求。

另一个观察是，ResNet 论文中 He 等人做过随机丢弃残差块（stochastic depth）的实验，发现训练时随机丢弃 30%–50% 的 block 网络仍能正常训练，部分配置下结果更优。这说明 ResNet 中**存在较多冗余层**——它们的输出仅贡献少量修正，去除后影响不大。这表明深层网络的瓶颈不在"层不够深"，而在"特征复用不充分"——每个 block 学到的内容只在紧接着的一两个 block 里被使用一次，之后就被加法叠到总和中。

2016 年下半年，社区开始讨论："ResNet 是否存在容量浪费？是否应当尝试新的连接方式？"

## 核心思想

DenseNet 的方案简单直接：**用 concat 替代 add**。在一个 Dense block 内，第 ℓ 层不再只接收前一层的输出，而是接收前面**所有 ℓ-1 层**的输出拼接（concat）成的特征图：

$$
x_\ell = H_\ell([x_0, x_1, \ldots, x_{\ell-1}])
$$

这里 $[\cdot]$ 表示沿通道维度的拼接，$H_\ell$ 是"BN + ReLU + 3×3 Conv"组合（pre-activation 风格，受 He 2016 *Identity Mappings* 启发）。每层的输出**永远不会被覆盖、不会被叠加、不会被混叠**——它就摆在那儿，谁想用谁拿。

```mermaid
graph TD
    x0["x₀: Input [B,k₀,H,W]"]:::input
    h1["H₁: BN-ReLU-Conv3×3"]:::compute
    h2["H₂: BN-ReLU-Conv3×3"]:::compute
    h3["H₃: BN-ReLU-Conv3×3"]:::compute
    h4["H₄: BN-ReLU-Conv3×3"]:::compute
    h5["H₅: BN-ReLU-Conv3×3"]:::compute
    out["Output [B, k₀+5k, H, W]"]:::output

    x0 --> h1
    x0 --> h2
    x0 --> h3
    x0 --> h4
    x0 --> h5
    h1 --> h2
    h1 --> h3
    h1 --> h4
    h1 --> h5
    h2 --> h3
    h2 --> h4
    h2 --> h5
    h3 --> h4
    h3 --> h5
    h4 --> h5
    h1 --> out
    h2 --> out
    h3 --> out
    h4 --> out
    h5 --> out

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```
*图 1：Dense block 内部稠密连接——第 ℓ 层 concat 前面所有层的输出作为输入，每层贡献 k 个新通道。*

**Growth rate $k$ 是 DenseNet 的关键尺度变量**。每层 $H_\ell$ 只产生 $k$ 个新通道（典型 $k = 32$），因此即使一个 Dense block 含 12 层、每层都接收前面全部输出，整条链上的通道数也只是 $k_0 + (\ell-1) \cdot k$ 的线性增长，不会指数膨胀。Growth rate 控制每层新增的通道数量，已有特征则通过 concat 被后续层复用。

整个 DenseNet 把网络切成**若干 Dense block**，block 内部稠密连接、block 之间用 **transition layer（1×1 Conv + 2×2 AvgPool）** 做下采样并压缩通道数（DenseNet-BC 版本里 transition 还会把通道数减半，进一步控制规模）。主流变体 DenseNet-121 / 169 / 201 / 264，数字指总有参层数；DenseNet-121 是最常用基线，4 个 Dense block 分别含 6 / 12 / 24 / 16 层，growth rate $k=32$。

```mermaid
graph TD
    in["Input [B,3,224,224]"]:::input
    stem["Conv 7×7 / s=2 / 64 + MaxPool"]:::compute
    db1["Dense Block 1 × 6 (k=32)"]:::compute
    t1["Transition: 1×1 Conv + AvgPool"]:::compute
    db2["Dense Block 2 × 12 (k=32)"]:::compute
    t2["Transition: 1×1 Conv + AvgPool"]:::compute
    db3["Dense Block 3 × 24 (k=32)"]:::compute
    t3["Transition: 1×1 Conv + AvgPool"]:::compute
    db4["Dense Block 4 × 16 (k=32)"]:::compute
    gap["Global Avg Pool → [B,1024]"]:::compute
    fc["FC 1000"]:::compute
    out["Softmax [B,1000]"]:::output

    in --> stem --> db1 --> t1 --> db2 --> t2 --> db3 --> t3 --> db4 --> gap --> fc --> out

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```
*图 2：DenseNet-121 整体结构——stem + 4 个 Dense block (6, 12, 24, 16) 通过 transition layer 串联，最后 GAP + FC。*

**特征复用与隐式深度监督**——把所有前层 concat 进来这件事带来两个一起到场的好处。第一，浅层学到的低阶特征（边缘、纹理）可以被任何一个深层直接拿来用，不必经过中间层的层层加法稀释；浅层与深层之间有**直接通路**，深层不需要重新发明边缘检测器。第二，梯度从 loss 流回去时也走同一条 concat 路径——loss 对 $H_1$ 输出的导数等于所有后续层（$H_2, H_3, \ldots, H_L$）对 $H_1$ 输出依赖项的导数之和。这条结构让浅层永远能拿到"来自所有深层的多份监督信号"，相当于在每一层都隐式架了一个深度监督头。**ResNet 给梯度一条高速公路，DenseNet 给每层一束高速公路**——继承了 ResNet 那个 `+1` 的本质，只是把"加"换成"拼"。

DenseNet 在 ImageNet 上的结果体现了该路线的核心优势：**DenseNet-121 用 7M 参数**可接近 ResNet-50（25.6M 参数）的精度（Top-5 ~6%），参数效率约为 ResNet 的数倍。该工作获 CVPR 2017 Best Paper——继 ResNet 之后，视觉社区连续两届将最高奖授予以连接方式为核心创新的工作。

## 训练细节

| 维度 | 值 |
|---|---|
| 优化器 | SGD + Momentum |
| 学习率 | 0.1，在第 50% 和 75% 训练 epoch 各除以 10 |
| 动量 | 0.9 |
| 权重衰减 | 1×10⁻⁴ |
| Batch size | 256 |
| Epochs | 300（CIFAR）/ 90（ImageNet） |
| Dropout | 不用（BN 自带正则；CIFAR 无增强版本用 0.2） |
| 权重初始化 | He 初始化 $\mathcal{N}(0, 2/n)$ |
| 数据增强 | 短边随机缩放 + 224 随机裁剪 + 水平翻转 + 颜色扰动 |

**$H_\ell$ 内部顺序**——DenseNet 用 **BN → ReLU → 3×3 Conv** 这条 pre-activation 顺序（受 He 2016 *Identity Mappings in Deep Residual Networks* 启发），不是 ResNet 原版的 Conv-BN-ReLU。在 concat 拓扑下，先 BN 再过卷积比反过来更稳，因为前面每一层的输出通道都被拼了进来，量级可能不一致，BN 在卷积前先把每路统一到零均值单位方差。

**DenseNet-BC（Bottleneck + Compression）** 是论文里更高效的变体，所有 DenseNet-121/169/201 的官方实现都是 BC 版：

- **Bottleneck**：每个 $H_\ell$ 在 3×3 Conv 之前先加一个 **1×1 Conv 把输入压到 4k 通道**——concat 进来的输入通道数会随层数线性增长（block 末尾 $k_0 + (\ell-1) \cdot k$ 可能上千），1×1 先压一刀避免 3×3 在巨大输入通道上算
- **Compression**：transition layer 上的 1×1 Conv 把输出通道数减半（$\theta = 0.5$），进一步控制 block 之间的通道膨胀

这套组合把 DenseNet-121 的参数量压到 7.0M（DenseNet-169 14.1M、DenseNet-201 20.0M、DenseNet-264 33.3M）——同期 ResNet-50 是 25.6M、ResNet-152 是 60M。

**训练资源**：4 块 Tesla K40 GPU 并行；DenseNet-121 在 ImageNet 上训练约 1 周。

**显存代价**——这是 DenseNet 工程上的主要挑战。Concat 让 Dense block 内部第 ℓ 层的输入通道数线性增长到 $k_0 + (\ell-1) \cdot k$。数值看似不算多（DenseNet-121 第三个 block 最后一层输入约 $256 + 23 \times 32 = 992$ 通道），但**问题主要在显存而非计算量**：

- ResNet 一个 block 算完后，前一个 block 的中间激活可以释放
- DenseNet 一个 block 内**所有前层的中间激活都需保留在显存中**——后续层 concat 时需要它们做反向传播
- 朴素实现下 DenseNet-121 训练时显存占用比 ResNet-50 多 2–3 倍

2017 年原始实现因此在相同 batch size 下显存压力较大。工业界后续采用 **memory-efficient DenseNet**（NVIDIA 实现，反向时重算中间激活）回收了部分显存，代价是训练时间增加约 15%。**这也是为什么尽管 DenseNet-121 参数效率更高，工业界的视觉 backbone 默认仍是 ResNet-50**——后者"算完即释放"的特性对部署、多机训练以及 detection / segmentation 等显存密集的下游任务更友好。

**ImageNet 错误率（Top-5）：**

| 模型 | 参数量 | Top-5 错误率 |
|---|---|---|
| ResNet-50 | 25.6M | 6.7% |
| ResNet-152 | 60.2M | 5.6% |
| DenseNet-121 | 7.0M | 6.1% |
| DenseNet-169 | 14.1M | 5.5% |
| DenseNet-201 | 20.0M | 5.2% |
| DenseNet-264 | 33.3M | 5.0% |

**DenseNet-201 用 20M 参数到了 ResNet-152（60M）同档精度**——这就是 CVPR 2017 Best Paper 的核心论据。

## 关键代码

下面这段实现 DenseNet 的核心砖：单层 `DenseLayer`（BN-ReLU-1×1-BN-ReLU-3×3 + concat）和一个 `DenseBlock`（堆 N 个 DenseLayer，每层都把自己的输出拼回输入）。注意 `torch.cat` 那一行——这就是 DenseNet 与 ResNet 在代码上的唯一差别：

```python
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    """BC 版 H_ℓ：1×1 压到 4k → 3×3 出 k 通道，输出与输入 concat 而非 add。"""
    def __init__(self, in_c: int, growth_rate: int = 32, bn_size: int = 4):
        super().__init__()
        inter_c = bn_size * growth_rate  # 1×1 压到 4k
        self.bn1   = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, inter_c, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(inter_c)
        self.conv2 = nn.Conv2d(inter_c, growth_rate, 3, padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_c, H, W]，in_c = k₀ + (ℓ-1)·k
        out = self.conv1(self.relu(self.bn1(x)))      # [B, 4k, H, W]
        out = self.conv2(self.relu(self.bn2(out)))    # [B, k, H, W]
        return torch.cat([x, out], dim=1)             # ← 核心一行：concat 而非 add
                                                       # 输出 [B, in_c + k, H, W]

class DenseBlock(nn.Module):
    """堆 num_layers 个 DenseLayer，输入通道随层数线性增长。"""
    def __init__(self, num_layers: int, in_c: int, growth_rate: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_c + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)   # 每过一层通道数 += growth_rate
        return x           # 输出 [B, in_c + num_layers·k, H, W]
```

`torch.cat([x, out], dim=1)` 与 ResNet 的 `out + identity` 看起来只差一个字符，但拓扑性质完全不同——前者无损保留每层输出，后者把所有层叠成一个张量。DenseNet 的全部精神就在这一行里。

## 影响 / 后续

DenseNet 对后续视觉架构的影响分为两部分：一部分被广泛采用，另一部分在工程实践中较少使用。

**被采用的部分是"特征复用"这一思路**——后续多数架构都把"如何让浅层特征被深层直接使用"作为显式的设计变量。U-Net 的 encoder-decoder skip、Feature Pyramid Network（FPN）的多尺度融合、HRNet 在所有分辨率上并行保持表征，都可视为 DenseNet 思想在不同尺度上的变体：信息不被中间层覆盖，浅层与深层之间保持直接通路。Transformer 时代的 [ViT](../08-vit/) 虽然不用 concat，但每个 block 中的 residual 加上 token 级全连接注意力本质上达成了类似效果——任何位置和深度的特征都能被全网访问。

**工程上较少使用的部分是"concat 连接方式本身"**——工业界更常选择 ResNet 而非 DenseNet 的原因主要不在精度，而在 **显存代价与推理友好性**。ResNet 的 add 让每个 block 算完后前面的中间激活可立即释放；DenseNet 的 concat 在训练时需保留全部前层激活，部署时也面对 channel 数线性增长带来的 memory bandwidth 问题。在同等精度下，工业部署多数选择 ResNet 系——这也是 Faster R-CNN / Mask R-CNN / DeepLab / CLIP 视觉塔的默认 backbone 为 ResNet-50/101 而非 DenseNet-121 的原因。**架构选择不仅看精度，也要考虑显存开销**——DenseNet 是这一权衡的一个典型例子。

EfficientNet 在 2019 年把这件事推进一步：与其在"加法 vs 拼接"里二选一，不如把 depth / width / resolution 三轴系统化地缩放，让架构搜索本身决定每个尺度上哪种连接更划算。

→ [05-resnet.md](05-resnet.md) · 加法残差 vs 拼接残差，两种思路最直接对照
→ [07-efficientnet.md](07-efficientnet.md) · 把"复用 vs 加法 vs 拼接"放进三轴缩放体系
→ [../foundations/04-normalization/](../foundations/04-normalization/) · DenseNet 用 pre-activation BN-ReLU-Conv 顺序，BN 在 concat 后统一各路量级
→ [../foundations/05-initialization/](../foundations/05-initialization/) · He 初始化让 DenseNet 也能从随机初始化直接开训
