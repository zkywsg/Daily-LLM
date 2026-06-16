---
name: "DCGAN"
year: 2015
family: "04-gan"
order: 2
paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
authors: ["Alec Radford", "Luke Metz", "Soumith Chintala"]
key_idea: "把 CNN 完整移植到 GAN——用 strided conv 替代 pooling、加 BatchNorm、Generator 用 transpose conv 上采样、去全连接层;首次给出可复现的 GAN 训练工程方案,生成 64×64 卧室 / 人脸图像"
---

## 前作进展

[GAN](01-gan.md) 2014 年发表后,理论很优雅但**工程极其脆弱**:

**1. 训练不稳定** —— 大半实验崩溃。D 损失迅速下降到 0,G 学不到东西;或 G 损失爆炸,生成噪声

**2. Mode Collapse 频发** —— G 只生成几种固定 pattern。MNIST 上 G 可能只生成数字 "1",其他数字一个都没

**3. 全连接层效率低** —— Goodfellow 2014 用 MLP 做 G/D,在 32×32 以上图像直接 OOM 或不收敛

**4. 没有可复现方案** —— 不同论文用的超参数千差万别,社区一直没有"GAN 标准训练食谱"

2014-2015 年大量人尝试改进 GAN 但都失败。直到 Radford 等人(Indico,2015 年 11 月)发现:**把 GAN 完全用 CNN 替代 MLP,配合几条工程规则,GAN 能稳定训练**。

DCGAN 论文标题虽然是"Unsupervised Representation Learning"(强调无监督表示学习),但真正影响是它给出了 **GAN 训练食谱**——之后几年所有 GAN 工作的基础设施。

论文的影响立竿见影:

- 论文里的 latent 空间算术(eye-glasses man - man + woman = eye-glasses woman)成 AI 圈火爆 meme
- 64×64 卧室 / 人脸生成质量震撼,让生成模型第一次"出圈"
- Soumith Chintala(论文三作)后来成为 PyTorch 创始人,DCGAN 也成为 PyTorch 教程经典案例

**没有 DCGAN 就没有后续 GAN 黄金时代**——CycleGAN、StyleGAN 等都建立在 DCGAN 的 CNN-GAN 范式上。

## 核心思想:CNN-GAN 工程食谱

DCGAN 的贡献不在数学,而在工程——一组具体的架构指南让 GAN 训练稳定。论文总结为几条规则:

### 规则 1: 用 strided conv / transpose conv 替代所有 pooling

```
# 不好(原始 CNN):
Conv → Pool → Conv → Pool → ...

# DCGAN:
D 用 strided Conv 下采样:Conv(stride=2) → Conv(stride=2) → ...
G 用 transposed Conv 上采样:ConvT(stride=2) → ConvT(stride=2) → ...
```

为什么?Pooling 是固定的(不可学),strided conv 让网络自己学下/上采样方式。

### 规则 2: G 和 D 都用 BatchNorm(除了 D 的输入层和 G 的输出层)

BatchNorm 让训练稳定,减少 mode collapse。这是 GAN 训练成功的关键 trick。

### 规则 3: 去掉全连接层

更深的 CNN,直接 conv 到 1×1 输出(D)或从 1×1×100 噪声 conv 出图像(G)。

### 规则 4: G 用 ReLU(输出层用 Tanh),D 用 LeakyReLU(0.2)

- G 内部 ReLU 让梯度流通
- G 输出用 Tanh,把图像 normalize 到 [-1, 1]
- D 用 LeakyReLU 避免 dying ReLU

### Generator 结构(64×64 输出)

```
z (100-d 噪声)
  ↓ reshape
1 × 1 × 100
  ↓ ConvT(stride=1)  → BatchNorm → ReLU
4 × 4 × 1024
  ↓ ConvT(stride=2)  → BatchNorm → ReLU
8 × 8 × 512
  ↓ ConvT(stride=2)  → BatchNorm → ReLU
16 × 16 × 256
  ↓ ConvT(stride=2)  → BatchNorm → ReLU
32 × 32 × 128
  ↓ ConvT(stride=2)  → Tanh
64 × 64 × 3  ← 输出图像
```

### Discriminator 结构(64×64 输入)

```
64 × 64 × 3  ← 输入图像
  ↓ Conv(stride=2)   → LeakyReLU
32 × 32 × 128
  ↓ Conv(stride=2)   → BatchNorm → LeakyReLU
16 × 16 × 256
  ↓ Conv(stride=2)   → BatchNorm → LeakyReLU
8 × 8 × 512
  ↓ Conv(stride=2)   → BatchNorm → LeakyReLU
4 × 4 × 1024
  ↓ Conv(stride=1)   → Sigmoid
1 × 1 × 1  ← 是否真
```

D 与 G **结构对称**——一个把图像 conv 到 scalar,另一个把 scalar 反 conv 到图像。这种对称设计成后续 GAN 标配。

### 训练超参数

- 优化器:**Adam(lr=2e-4, β₁=0.5, β₂=0.999)** —— β₁ 从默认 0.9 改到 0.5 是关键,避免 momentum 把 G/D 推向极端
- batch size = 128
- 用 LeakyReLU 0.2 斜率

这些超参数后来成为 GAN 训练默认配置。

## 关键代码

DCGAN Generator + Discriminator(PyTorch):

```python
import torch
import torch.nn as nn

class DCGAN_Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: z, (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # (ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf, 32, 32)
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # (img_channels, 64, 64)
        )

    def forward(self, z):
        return self.net(z)


class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # (img_channels, 64, 64)
            nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 32, 32)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 16, 16)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 8, 8)
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 4, 4)
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# 权重初始化(论文推荐)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = DCGAN_Generator().cuda()
D = DCGAN_Discriminator().cuda()
G.apply(weights_init)
D.apply(weights_init)

# 训练用 Adam(lr=2e-4, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
```

## 性能数据

DCGAN 没有用 IS / FID(那时还没发明)。论文用几种定性 / 间接评估:

### 1. CIFAR-10 监督学习(用 D 作为特征提取器)

把训完的 DCGAN 的 D 当 feature extractor,接一个 SVM 做 CIFAR-10 分类:

| Model | CIFAR-10 Acc |
|------|------|
| K-means(unsupervised baseline)| 80.6 |
| **DCGAN + L2-SVM** | **82.8** |
| Supervised CNN | 84.8 |

DCGAN 学到的特征接近监督学习水平,证明 GAN 训练学到了有意义的表示。

### 2. Latent space arithmetic

最著名的"DCGAN demo":

```
G(z_man_with_glasses) - G(z_man) + G(z_woman) ≈ G(z_woman_with_glasses)
```

把 latent 向量做算术,能得到语义合成的结果。这一现象第一次让人直观看到 **G 学到了语义结构化的 latent 空间**,GAN 不只是"乱画",而是真的"理解"了图像分布。

### 3. 卧室生成质量

LSUN bedroom 数据集 64×64 生成:

- DCGAN 1 epoch 生成:模糊但有"床 / 窗 / 灯"轮廓
- DCGAN 5 epoch:细节丰富,几乎能看出真实卧室
- 论文展示的 256 张生成卧室排在一起,大部分让人难以分辨真假

### 4. 人脸生成

CelebA 人脸:64×64 生成质量震撼,虽然分辨率低但五官、头发、表情都自然。

## 影响 / 后续

DCGAN 在 GAN 历史的位置:**GAN 工程化的奠基,后续所有 GAN 工作的基础设施**。

**1. 标准 GAN 训练食谱** —— DCGAN 的几条规则(strided conv、BatchNorm、Adam lr=2e-4 β₁=0.5)成为 2015-2018 年所有 GAN 工作的默认配置。即使 WGAN / SAGAN 等改进 GAN,核心结构仍基于 DCGAN

**2. GAN 训练可复现性大幅提升** —— DCGAN 之前 GAN 训练成功率 < 30%,DCGAN 之后接近 90%+。这一可复现性让大量研究者能进入 GAN 领域

**3. Latent space 算术成 GAN 标志性现象** —— DCGAN 的人脸 latent 算术后来被 InfoGAN / BiGAN / StyleGAN 等系统化,成为 GAN 可解释性研究的核心方向

**4. 直接催生后续大量 GAN 变体** —— Improved GAN(2016)、CoGAN(2016)、SeqGAN(2017)、pix2pix(2017)、CycleGAN(2017)等都用 DCGAN 风格的 CNN 架构

**5. Soumith Chintala 与 PyTorch 的渊源** —— DCGAN 三作 Soumith 后来在 Facebook 主导 PyTorch 开发。PyTorch 教程里 DCGAN 是经典 GAN 案例,影响了一代 PyTorch 学习者

**6. 学术影响延续到今天** —— 即使在 Diffusion 时代,SD 的 VAE encoder 仍受 DCGAN-style 卷积结构启发。U-Net 也借鉴了 G 和 D 的对称设计

DCGAN 留下的开放问题:

- **质量上限** —— DCGAN 只能稳定到 64×64,128×128 经常 mode collapse → Progressive GAN 解决
- **训练仍偶发不稳** —— BatchNorm + Adam 帮助大但不彻底 → WGAN / Spectral Norm 进一步解决
- **mode collapse 没根治** —— DCGAN 还是会塌缩到部分模式 → minibatch discrimination / unrolled GAN
- **可控性差** —— G(z) 给随机噪声出随机图,没法指定生成"戴眼镜的男人" → [Conditional GAN](https://arxiv.org/abs/1411.1784) / [CycleGAN](03-cyclegan.md)

→ [03-cyclegan.md](03-cyclegan.md) · 应用层突破,GAN 进入艺术 / 风格迁移
→ [04-stylegan.md](04-stylegan.md) · 质量飞跃,1024² 高分辨率人脸
→ [01-gan.md](01-gan.md) · 数学起源,DCGAN 是其工程化版本
→ [../01-cnn/](../01-cnn/) · DCGAN 把 CNN 系统性移植到 GAN
→ [../foundations/](../foundations/) · BatchNorm、Adam 是 DCGAN 训练成功的关键
