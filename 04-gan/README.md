# GAN 生成对抗

> **让两个网络互相博弈——一个造假,一个鉴别,把"生成"问题转化为"对抗"问题,开启深度生成模型的第一个黄金时代。**

## 一句话定位

这家族解决的是深度学习时代之前的一个根本难题——**怎么让神经网络生成真实的图像 / 音频 / 文本**。2014 年之前的生成模型(VAE / RBM / autoregressive)都受限于"显式建模 likelihood"的数学框架,生成的图片模糊、缺乏细节。2014 年 Goodfellow 等人的 **GAN**(Generative Adversarial Network)给出完全不同的思路:**不显式建模 likelihood,而是让生成器 G 与判别器 D 对抗博弈**——G 尽力造假骗 D,D 尽力分辨真假,纳什均衡时 G 学到真实数据分布。2015 年 **DCGAN**(Radford)第一次让 GAN 训练稳定——用 CNN 替代全连接,加入 BatchNorm 等工程 trick,生成 64×64 卧室 / 人脸图像,开启 GAN 的工程化时代。2017 年 **CycleGAN**(Zhu)用 cycle consistency loss 实现**无配对图像翻译**——把马变斑马、夏天变冬天、照片变莫奈画风,不需要成对训练数据,引爆 GAN 在艺术 / 风格迁移领域的爆发。2018 年 **StyleGAN**(Karras,NVIDIA)用 style-based generator 生成 1024×1024 超高分辨率人脸,质量逼近真实照片,生成的"this person does not exist"成为 GAN 文化代表。这家族要回答的问题是:**深度学习时代如何用对抗博弈生成高质量内容,从理论 demo 到 4K 人脸的 4 年演化**。

## 概念本身

GAN 的核心思想是**对抗博弈**(adversarial game)。两个网络互相博弈:

- **生成器 G**(Generator):输入随机噪声 z,输出假数据 G(z)
- **判别器 D**(Discriminator):输入真数据 x 或假数据 G(z),输出"是真"的概率

训练目标是 minimax 博弈:

$$
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

- D 想最大化:正确判断真假
- G 想最小化:让 D 把 G(z) 判为真

理论上,当博弈达到 Nash equilibrium 时,G 学到的分布 = 真实数据分布,D 输出 0.5(完全分不清)。

### 与其他生成模型的关键差异

| 模型 | 优化目标 | 优势 | 短板 |
|------|------|------|------|
| **VAE** | likelihood + KL | 训练稳定 / 可推理 z | 模糊(L2 loss 平均化) |
| **Autoregressive(PixelRNN)** | log-likelihood | 精确 likelihood | 慢(逐像素) |
| **GAN** | adversarial | **图像清晰** | 训练不稳定 / mode collapse |
| **Diffusion(后来)** | denoising | 质量 + 可控 | 推理慢 |

GAN 在 2014-2020 是图像生成的绝对主流,直到 2020 年 [Diffusion](../10-diffusion/01-ddpm.md) 兴起才被部分替代。今天 GAN 仍在风格迁移、超分辨率、艺术创作等领域活跃。

### GAN 的几个核心问题

**1. Mode Collapse(模式塌缩)** —— G 只学到数据分布的一部分,生成的样本多样性低。比如 MNIST 上只生成数字"3"。这是 GAN 最经典的训练失败模式

**2. 训练不稳定** —— D 和 G 的能力不平衡会导致一方碾压。D 太强 → G 没梯度;G 太强 → D 学不到东西。需要精细调参

**3. 评估困难** —— 没有 likelihood,怎么量化"生成质量"?Inception Score / FID 等代理指标被发明出来

**4. 难收敛** —— minimax 博弈在非凸优化下没有理论保证收敛,需要大量工程 trick

### 几条主要演化主线

- **稳定训练**:GAN(2014)→ DCGAN(2015)→ WGAN(2017)→ Spectral Norm GAN(2018)
- **应用扩展**:Image2Image(pix2pix)→ CycleGAN(无配对)→ StarGAN(多域)
- **质量提升**:DCGAN(64²)→ Progressive GAN(1024²)→ StyleGAN(高质量)→ StyleGAN3(无 alias)
- **特殊任务**:SRGAN(超分)、CycleGAN(风格迁移)、Pix2PixHD(高清)、BigGAN(条件生成)

理解 GAN 家族 = 理解 2014-2020 深度生成模型的黄金时代,以及今天 diffusion / SD 时代的前史。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2014 | **GAN** | Minimax 对抗博弈框架,G 与 D 互相训练;不显式建模 likelihood 也能学到数据分布;深度生成模型新范式 | VAE 生成模糊,autoregressive 慢,显式 likelihood 路线难以生成高质量图像 |
| 2015 | **DCGAN** | 把 CNN 用到 GAN(替代全连接),加 BatchNorm / 去 pooling,首次稳定训练;64×64 卧室 / 人脸生成 | 原始 GAN 训练极不稳定,大半实验崩溃,缺乏可复现工程方案 |
| 2017 | **CycleGAN** | Cycle consistency loss 实现无配对图像翻译,马→斑马 / 夏→冬 / 照片→画风;GAN 在艺术领域爆发 | 之前 pix2pix 需要成对数据(同场景两张图),许多任务无法获得 |
| 2018 | **StyleGAN** | Style-based generator,把 z 映射到 W 空间再注入每层,实现风格 disentanglement;1024² 超高分辨率人脸,"this person does not exist" 文化现象 | Progressive GAN 能高分辨率但风格不可控,latent 空间缠绕,无法精细控制生成 |

## 依赖与延伸

**前置(foundations):**
- [../01-cnn/01-lenet.md](../01-cnn/01-lenet.md) —— DCGAN 把 CNN 用到 GAN 上,需要 CNN 基础
- [../01-cnn/05-resnet.md](../01-cnn/05-resnet.md) —— Progressive GAN / StyleGAN 用到 residual 思想
- [../foundations/](../foundations/) —— BatchNorm、Adam 优化器等 GAN 训练必备组件

**通向哪些家族:**
- [../10-diffusion/](../10-diffusion/) —— Diffusion 在 2020 年后部分取代 GAN,但概念上互补
- [../09-multimodal-clip/](../09-multimodal-clip/) —— GAN 也曾用于 text-to-image(StackGAN / AttnGAN),后被 CLIP+Diffusion 替代
- [../08-vit/04-dit.md](../08-vit/04-dit.md) —— Diffusion Transformer 是 GAN → Diffusion 路线的延续
