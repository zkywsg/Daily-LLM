---
name: "GAN"
year: 2014
family: "04-gan"
order: 1
paper: "Generative Adversarial Networks"
authors: ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza", "Bing Xu", "David Warde-Farley", "Sherjil Ozair", "Aaron Courville", "Yoshua Bengio"]
key_idea: "把生成问题转化为对抗博弈——G 造假,D 鉴别,minimax 训练让 G 学到真实数据分布;不显式建模 likelihood 也能生成高质量样本,开创深度生成模型新范式"
---

## 前作进展

2014 年之前,深度生成模型主要有几条路:

**1. RBM / DBN**(Hinton 2006)—— Restricted Boltzmann Machine 是早期深度学习的代表,但需要 MCMC 采样,生成慢且质量差

**2. VAE**(Kingma 2013,与 GAN 同年)—— Variational Autoencoder 通过最大化 ELBO 训练,有理论保证但生成的图像**模糊**(L2 reconstruction loss 把所有可能模式平均化)

**3. Autoregressive**(PixelRNN 2016 / 早期 NADE)—— 逐像素生成,精确建模 likelihood 但生成极慢

**4. Energy-based Models** —— 理论优雅但训练不稳定,partition function 难算

所有这些方法的共同问题:**显式建模 likelihood 或 partition function 太难,生成图像受限于 loss 函数的"平均化"特性**。

Goodfellow 等人(Bengio 组,2014 年 6 月)的 GAN 论文给出完全不同的思路:**根本不显式建模 likelihood,通过对抗训练让 G 隐式学到数据分布**。这一思路有几个革命性优势:

- **不需要 partition function** —— 完全避开 likelihood 计算的数学困难
- **G 可以是任意 differentiable 函数** —— 任意神经网络都可以做 G
- **生成清晰** —— 没有 L2 平均化,生成的图像有细节

Goodfellow 在论文里讲了一个传说级故事:他在蒙特利尔的一家酒吧 brainstorm 时想出 GAN 的对抗博弈思路,当晚回家就实现并跑通了。从想法到 paper 投稿仅几周。**GAN 是深度学习历史上最有"a-ha moment"特征的工作之一**。

## 核心思想:Minimax 对抗博弈

### 数学公式

GAN 的目标函数是一个 minimax 博弈:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

- $x \sim p_{\text{data}}$ —— 真实数据样本
- $z \sim p_z$ —— 随机噪声(通常是 $\mathcal{N}(0, I)$)
- $D(x) \in [0, 1]$ —— D 判断 x 为真的概率
- $G(z)$ —— G 用噪声 z 生成的假样本

D 想最大化 V:对真样本输出 1,对假样本输出 0。G 想最小化 V:让 D 把 G(z) 也判为真(即 D(G(z)) ≈ 1)。

### 理论保证

论文证明了一个漂亮的理论结果:

**1. 给定 G,最优 D 是:**

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}
$$

**2. 代入回 V,得到 G 的等价优化目标:**

$$
C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_G)
$$

其中 JSD 是 Jensen-Shannon Divergence(对称的 KL)。

**3. 全局最优在 $p_G = p_{\text{data}}$ 处取得**,此时 JSD = 0,C(G) = -log 4 ≈ -1.386,D(x) = 1/2(完全分不清真假)。

这是 GAN 的理论基石——只要训练能收敛,G 就学到真实数据分布。**问题在于"训练能收敛"这一前提**——后续大量工作就是处理 GAN 训练不稳定。

### 实际训练:Alternating Update

理论是 minimax,实际训练是交替更新:

```
for each iteration:
    # Step 1: 训练 D
    采样真数据 x, 假数据 G(z)
    更新 D 最大化:log D(x) + log(1 - D(G(z)))

    # Step 2: 训练 G
    采样新的噪声 z
    更新 G 最小化:log(1 - D(G(z)))
    # 实际用 -log D(G(z))(non-saturating loss,梯度更稳)
```

**Non-saturating loss trick**:用 `-log D(G(z))` 替代 `log(1 - D(G(z)))`,在训练早期 D 强 G 弱时梯度更稳。

### G 与 D 的平衡

GAN 训练有个微妙问题:**D 不能太强,否则 G 没梯度**。论文里一些经验:

- D 训练 k 步,G 训练 1 步(k 通常 1 或 5)
- 用 mini-batch SGD
- 用 momentum

但这些超参数调起来很难,GAN 训练失败率在 2014 年极高。后续 [DCGAN](02-dcgan.md) 才给出可靠的工程方案。

## 关键代码

PyTorch 实现最简 GAN(MLP-based,跑 MNIST):

```python
import torch
import torch.nn as nn

# 1. Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh(),  # 输出 [-1, 1]
        )
    def forward(self, z):
        return self.net(z)

# 2. Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# 3. Training loop
G = Generator().cuda()
D = Discriminator().cuda()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(200):
    for real_imgs, _ in mnist_loader:
        bs = real_imgs.size(0)
        real_imgs = real_imgs.view(bs, -1).cuda()
        real_labels = torch.ones(bs, 1).cuda()
        fake_labels = torch.zeros(bs, 1).cuda()

        # Train D
        opt_D.zero_grad()
        real_loss = criterion(D(real_imgs), real_labels)
        z = torch.randn(bs, 100).cuda()
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_D.step()

        # Train G(non-saturating loss)
        opt_G.zero_grad()
        z = torch.randn(bs, 100).cuda()
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)  # 让 D 判为真
        g_loss.backward()
        opt_G.step()
```

跑通 MNIST 大概要 200 epoch,生成的数字勉强能看(2014 年水平)。

## 性能数据

GAN 原论文没有定量 benchmark(2014 年还没有 IS / FID 这些指标)。论文用 MNIST、Toronto Face Database(TFD)、CIFAR-10 做定性 demo:

- **MNIST** —— 生成的数字模糊但能识别
- **TFD** —— 生成的人脸有基本五官但细节差
- **CIFAR-10** —— 生成的图像高度模糊(32×32 仍勉强)

**Parzen window log-likelihood**(GAN 时代主要的量化指标,粗糙):

| 模型 | MNIST | TFD |
|------|------|------|
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | 2110 ± 50 |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| **GAN** | **225 ± 2** | **2057 ± 26** |

GAN 在 MNIST 上最好,TFD 上接近 SOTA。但 Parzen window 不可靠,后来研究证明这个指标几乎和质量无关。

**真正让 GAN 出圈的是定性结果**——尤其在 [DCGAN](02-dcgan.md) 之后,GAN 生成的人脸 / 卧室让人直观看到"机器在创造图像"。

## 影响 / 后续

GAN 在深度学习历史的位置:**深度生成模型新范式,2014-2020 图像生成的绝对主流**。

**1. GAN 引爆生成模型研究** —— 2014-2018 年 GAN 论文数量爆发式增长,arXiv 上每周几十篇 GAN 变体(称为 "GAN Zoo"——SAGAN, BigGAN, ProGAN, ...)。Goodfellow 这一篇论文引用 60K+,是深度学习史上最高引用论文之一

**2. 直接催生 DCGAN / WGAN / CycleGAN 等里程碑** —— GAN 的工程化(DCGAN)、稳定化(WGAN)、应用化(CycleGAN)等后续工作都基于这个 minimax 框架

**3. Adversarial 思想扩散** —— Adversarial training 思路从生成模型扩散到其他领域:adversarial examples / adversarial robustness / adversarial domain adaptation 等。"对抗"成为深度学习核心概念之一

**4. 工业应用爆发** —— Deepfake、StyleGAN 生成的"this person does not exist"、艺术 GAN(GANbreeder, ArtBreeder)、超分辨率(SRGAN)、图像翻译(CycleGAN)等。GAN 第一次让"AI 创作内容"进入大众视野

**5. 启发 Diffusion** —— Diffusion 2020 年崛起后部分取代 GAN,但 Diffusion 设计上仍受 GAN 影响。Classifier-Free Guidance / ControlNet 等 Diffusion 控制思路与 GAN 的 conditional 路线一脉相承

**6. Goodfellow 个人成为 AI 圈代表人物** —— GAN 论文一作,2016 年出版《Deep Learning》教材,后任职 Google Brain / Apple ML / DeepMind。在 LeCun / Hinton / Bengio "深度学习三巨头"之外的下一代代表

GAN 留下的开放问题(由后续工作解答):

- **训练不稳定** —— 怎么让 GAN 训练可靠?→ [DCGAN](02-dcgan.md) 给出 CNN-GAN 工程方案,WGAN 用 Wasserstein 距离替代 JSD
- **Mode collapse** —— G 只生成数据分布的一部分?→ Unrolled GAN / minibatch discrimination
- **没法控制生成内容** —— G(z) 给随机噪声出随机图,怎么按要求生成?→ Conditional GAN / [CycleGAN](03-cyclegan.md) / [StyleGAN](04-stylegan.md)
- **质量上限** —— 怎么生成 1024×1024 高分辨率?→ Progressive GAN / [StyleGAN](04-stylegan.md)
- **评估难** —— 没有 likelihood 怎么量化质量?→ Inception Score / FID

→ [02-dcgan.md](02-dcgan.md) · CNN-GAN 工程化,首次稳定训练
→ [03-cyclegan.md](03-cyclegan.md) · 无配对图像翻译,GAN 应用爆发
→ [04-stylegan.md](04-stylegan.md) · 风格控制 + 高分辨率,GAN 巅峰
→ [../10-diffusion/01-ddpm.md](../10-diffusion/01-ddpm.md) · 后继生成模型路线,部分取代 GAN
→ [../01-cnn/](../01-cnn/) · DCGAN 之后 GAN 普遍基于 CNN
