# 深度学习与大模型演进时间线

> 每一个技术的出现，背后都有一个"不得不解决"的问题。
> 这条时间线不是论文列表，而是一部"被逼出来的历史"。

<img src="assets/timeline.svg" alt="深度学习与大模型演进地图" width="100%">

## 导航

| 年份 | 核心事件 | 年份 | 核心事件 |
|------|---------|------|---------|
| [2012](#2012) | AlexNet — 深度学习元年 | [2019](#2019) | GPT-2 + T5 — 规模的野心 |
| [2013](#2013) | Word2Vec + VAE — 表示学习觉醒 | [2020](#2020) | GPT-3 + Scaling Laws — 大力出奇迹 |
| [2014](#2014) | GAN + Seq2Seq + Attention + Adam | [2021](#2021) | CLIP + Codex + LoRA — 多模态与效率 |
| [2015](#2015) | ResNet + BN — 深度的解放 | [2022](#2022) | ChatGPT — AI 走进大众 |
| [2016](#2016) | AlphaGo — 强化学习登台 | [2023](#2023) | GPT-4 + LLaMA — 开源的反击 |
| [2017](#2017) | Transformer — 把 RNN 扔掉 | [2024](#2024) | MoE + 长上下文 + 推理模型 |
| [2018](#2018) | BERT + GPT-1 — 预训练时代 | [2025](#2025) | DeepSeek R1 — 开源追平 |

---

<a id="2012"></a>
## 2012 · AlexNet：一声炮响，旧世界终结

### 之前的世界

在 2012 年之前，计算机视觉领域的主流做法是**手工设计特征**。研究者们花几年时间设计 SIFT、HOG 这样的特征描述子，告诉计算机"什么是边缘"、"什么是纹理"，然后再接一个 SVM 来做分类。这套方法能用，但有一个硬伤：**它的上限取决于人类对"特征"的理解**。

ImageNet 大规模视觉识别挑战赛（ILSVRC）从 2010 年开始举办，每年最好的模型 Top-5 错误率在 25%–26% 徘徊，进步缓慢。大家普遍觉得这个问题"差不多到头了"。

### 发生了什么

2012 年，多伦多大学 Hinton 组的三个人——Alex Krizhevsky、Ilya Sutskever、Geoffrey Hinton——提交了一个叫 AlexNet 的模型，直接把 Top-5 错误率打到了 **15.3%**，比第二名低了将近 11 个百分点。

差距大到评委以为他们搞错了。

AlexNet 本质上是一个深度卷积神经网络，但有几个关键的工程决策让它跑起来：

- **ReLU 激活函数**：把 sigmoid/tanh 换掉，梯度不再消失，训练速度快 6 倍
- **GPU 并行训练**：用两张 GTX 580 分摊计算，首次证明 GPU 是训练深度网络的正确载体
- **Dropout**：训练时随机丢掉一半神经元，强迫网络学会鲁棒特征，不过拟合
- **数据增强**：随机裁剪、翻转，让 120 万张图片"变出"更多样本

### 解决了什么，又带来了什么新问题

AlexNet 证明了一件事：**特征不需要人来设计，网络可以自己学**。这一枪打穿了手工特征时代的天花板。

但它也带来了新问题：
- 需要**大量有标注的数据**（ImageNet 有 120 万张图，现实中哪里来这么多）
- 需要 **GPU**，当时大多数研究者根本没有
- 网络为什么有效，没有人能完全解释清楚（可解释性问题延续至今）

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| Dropout | Hinton 组（Toronto） | 随机失活防过拟合，深度学习标准正则化手段确立 | [01·视觉线](../01-Visual-Intelligence/) |
| ReLU 全面普及 | Hinton et al. | 取代 sigmoid/tanh，梯度流动恢复，训练速度提升数倍 | [00·前置](../00-Prerequisites/) |
| GPU 深度学习生态 | NVIDIA / Krizhevsky | CUDA 加速深度网络训练，计算基础设施范式确立 | [00·前置](../00-Prerequisites/) |
| ImageNet LSVRC | Stanford / Princeton | 建立深度学习时代最重要的视觉 Benchmark，推动年度竞争 | [00·前置](../00-Prerequisites/) |
| 深度学习语音识别 | Hinton + Google/MS | DNN 在 ASR 上首次大规模工业验证，错误率骤降 20%+ | [01·视觉线](../01-Visual-Intelligence/) |
| 数据增强技术 | Krizhevsky et al. | 随机裁剪 / 翻转 / 颜色抖动，少数据训练大模型的标准做法 | [00·前置](../00-Prerequisites/) |
| Max Pooling 标准化 | — | 空间降采样 + 平移不变性，CNN 架构的标准组件 | [01·视觉线](../01-Visual-Intelligence/) |
| Local Response Normalization | Krizhevsky | 横向抑制机制，BN 出现前的归一化手段 | [01·视觉线](../01-Visual-Intelligence/) |
| DBN → CNN 范式转移 | Hinton | 无监督预训练时代终结，端到端有监督训练成为主流 | [01·视觉线](../01-Visual-Intelligence/) |
| 卷积特征可视化研究 | — | 理解 CNN 各层在学什么，可解释性研究的起点 | [01·视觉线](../01-Visual-Intelligence/) |

---

<a id="2013"></a>
## 2013 · Word2Vec：词也能有坐标

### 之前的世界

自然语言处理有一个最基本的问题：怎么把词喂给模型？

当时的标准答案是 **One-Hot 编码**：词汇表里有 10 万个词，每个词就是一个 10 万维的向量，只有对应位置是 1，其余全是 0。这有两个严重问题：

1. **维度灾难**：10 万维的稀疏向量，计算极其低效
2. **没有语义关系**：在 One-Hot 的世界里，"猫"和"狗"的距离，跟"猫"和"飞机"的距离一模一样——模型根本不知道猫和狗更像

### 发生了什么

2013 年，Google 的 Tomas Mikolov 团队发表了 Word2Vec，提出了两种训练方式：

- **CBOW**（Continuous Bag of Words）：用上下文词预测中心词
- **Skip-gram**：用中心词预测上下文词

核心思想极其简单：**一个词的意思，由它的邻居决定**（语言学上叫"分布假说"）。训练完之后，每个词变成一个几百维的稠密向量。

这个向量有一个让人惊艳的性质：
`king - man + woman ≈ queen`

词向量的加减法，居然能反映语义关系。

### 解决了什么，又带来了什么新问题

Word2Vec 给 NLP 带来了"词的坐标系"，让语义相似的词在空间上靠近，下游任务的效果大幅提升。

但它有一个根本局限：**每个词只有一个向量**。"苹果"这个词，不管是水果还是公司，向量都一样。一词多义的问题，Word2Vec 无力解决——这个问题要等到 2018 年才有真正的答案。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| VAE（变分自编码器） | Kingma & Welling | 用变分推断学连续潜变量，生成模型的数学基础 | [01·视觉线](../01-Visual-Intelligence/) |
| ZFNet | Zeiler & Fergus（NYU） | 反卷积可视化 CNN，赢得 ILSVRC 2013，CNN 可解释性开端 | [01·视觉线](../01-Visual-Intelligence/) |
| Network in Network | Lin et al. | 1×1 卷积 + Global Avg Pooling，影响 Inception、ResNet 等所有后续架构 | [01·视觉线](../01-Visual-Intelligence/) |
| DropConnect | Wan et al.（NYU） | 在权重而非激活上随机置零，Dropout 的泛化变体 | [01·视觉线](../01-Visual-Intelligence/) |
| Maxout Networks | Goodfellow et al. | 分段线性激活，理论上可逼近任意凸函数，GAN 作者首篇代表作 | [01·视觉线](../01-Visual-Intelligence/) |
| DeepFace | Taigman et al.（Facebook） | 深度学习人脸识别首次超越人类水平（LFW 97.35%） | [01·视觉线](../01-Visual-Intelligence/) |
| LSTM 序列生成 | Graves | 用 LSTM 生成手写字迹，序列到序列生成能力的早期展示 | [01·视觉线](../01-Visual-Intelligence/) |
| DQN 前身 | Mnih et al.（DeepMind） | CNN + Q-learning 玩 Atari，深度强化学习的种子论文 | [00·前置](../00-Prerequisites/) |
| Negative Sampling | Mikolov et al. | Word2Vec 训练加速技巧，使大词汇表训练在 CPU 上可行 | [02·语言线](../02-Language-Transformers/) |
| GloVe 研究起步 | Pennington et al.（Stanford） | 全局共现矩阵词向量，2014 年正式发表 | [02·语言线](../02-Language-Transformers/) |

---

<a id="2014"></a>
## 2014 · GAN + Seq2Seq + Attention + Adam：一年四响

2014 年是个密集的年份，四个东西同时冒出来，每一个都影响深远。

### GAN：让网络学会"对抗"

Ian Goodfellow 在一次酒吧聊天后回家连夜写出了 GAN（生成对抗网络）的代码。

思路：训练两个网络，**生成器**负责造假，**判别器**负责识破。两者互相博弈，最终生成器学会造出以假乱真的样本。

这是生成式 AI 的第一块基石。从此，模型不只是能"分类"，还能"创造"。

### Seq2Seq：机器翻译的新范式

Sutskever、Vinyals、Le（均来自 Google）提出 Seq2Seq 架构：一个 LSTM 编码器把输入压缩成一个向量，一个 LSTM 解码器从这个向量生成输出。

这个架构第一次让端到端的机器翻译成为可能，不再需要对齐表、语法规则这些手工模块。

但它有一个致命弱点：**无论输入多长，都要压缩成一个固定大小的向量**。翻译一篇文章和翻译一个词用的是同一个"瓶子"。

### Attention：给解码器一双"回头看"的眼睛

同年，Bahdanau 等人提出注意力机制，直接针对 Seq2Seq 的信息瓶颈：

> 解码每个词的时候，不要只看那个固定向量，而是**让解码器自己决定该重点看编码器的哪些位置**。

这个想法听起来简单，但打开了一扇门——后来的 Transformer 完全建立在它的基础上。

### Adam：优化器的终结者

Kingma 和 Ba 提出 Adam 优化器，结合了动量和自适应学习率，几乎不需要调学习率就能跑得很好。此后几年，Adam 成为深度学习的默认优化器，几乎没有竞争者。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| VGGNet | Simonyan & Zisserman（Oxford） | 3×3 卷积均匀堆叠，证明深度 > 宽度，迁移学习经典 backbone | [01·视觉线](../01-Visual-Intelligence/) |
| GoogLeNet / Inception | Szegedy et al.（Google） | Inception 模块多尺度并行卷积，ILSVRC 2014 冠军 | [01·视觉线](../01-Visual-Intelligence/) |
| Adam 优化器 | Kingma & Ba（OpenAI/Amsterdam） | 自适应学习率 + 动量，几乎不需要调参即可收敛，成为深度学习默认优化器 | [01·视觉线](../01-Visual-Intelligence/) |
| GloVe | Pennington et al.（Stanford） | 全局共现矩阵词向量，与 Word2Vec 互补，下游任务效果更稳 | [02·语言线](../02-Language-Transformers/) |
| GRU | Cho et al.（Montreal） | 简化 LSTM，两个门替代三个，计算更高效，序列建模新选择 | [01·视觉线](../01-Visual-Intelligence/) |
| Dropout in RNNs | Zaremba et al.（NYU） | 解决 RNN 过拟合问题，让 LSTM 语言模型训练更稳定 | [01·视觉线](../01-Visual-Intelligence/) |
| Neural Turing Machine | Graves et al.（DeepMind） | 神经网络外挂可读写记忆，早期 Agent 思想和 RAG 雏形 | [05·系统生产](../05-Systems-Production/) |
| FaceNet | Schroff et al.（Google） | Triplet Loss 人脸嵌入，人脸识别工业大规模部署的技术基础 | [01·视觉线](../01-Visual-Intelligence/) |
| Deep Learning（书） | Bengio, Goodfellow, Courville | 深度学习领域第一本系统教材，奠定学科知识体系 | [00·前置](../00-Prerequisites/) |
| DCGAN 概念萌芽 | — | GAN 训练技巧积累，为 2015 年 DCGAN 做铺垫 | [01·视觉线](../01-Visual-Intelligence/) |

---

<a id="2015"></a>
## 2015 · ResNet + Batch Norm：深度的解放

### 之前的世界

有了 AlexNet，大家的直觉是：**网络越深，效果越好**。但现实很残酷——深度超过一定层数之后，训练效果反而变差，不是因为过拟合，而是训练根本收敛不了。

这个问题叫**退化问题**（Degradation Problem）：一个 56 层的网络，居然比 20 层的还差。

### 发生了什么

何凯明（Kaiming He）等人提出了 **残差网络（ResNet）**，思路优雅到令人拍案：

既然直接学映射 `H(x)` 很难，那就让网络学**残差** `F(x) = H(x) - x`，再加上一条**跳跃连接**把输入 `x` 直接加回去。

这样，哪怕网络什么都没学到，跳跃连接至少保证输出不比输入差——网络"退无可退"。

同年，Ioffe 和 Szegedy 提出 **Batch Normalization**：在每一层之后对激活值做归一化，让每层的输入分布稳定下来。它让训练速度提升了一个数量级，允许使用更高的学习率。

### 解决了什么，又带来了什么新问题

ResNet 解放了网络深度。152 层的 ResNet 在 ImageNet 上把 Top-5 错误率压到了 3.57%，已经**低于人类水平（约 5%）**。

计算机视觉的分类问题，从这一年起基本宣告解决。接下来的问题是：同样的思路，能不能用在语言上？

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| ResNet | He et al.（MSRA） | 152 层残差网络，Top-5 错误率 3.57% 低于人类，跳跃连接解决退化问题 | [01·视觉线](../01-Visual-Intelligence/) |
| Batch Normalization | Ioffe & Szegedy（Google） | 归一化每层输入分布，训练速度提升数量级，允许更高学习率，深度网络稳定训练的基础 | [01·视觉线](../01-Visual-Intelligence/) |
| Highway Networks | Srivastava et al.（IDSIA） | 门控跳跃连接，ResNet 的直接先驱，最早解决网络退化 | [01·视觉线](../01-Visual-Intelligence/) |
| DQN（Nature 版） | Mnih et al.（DeepMind） | 深度强化学习第一个重量级成果，Atari 57 款游戏超越人类 | [00·前置](../00-Prerequisites/) |
| Faster R-CNN | Ren et al.（MSRA） | 区域建议网络统一检测流程，目标检测标准范式确立 | [01·视觉线](../01-Visual-Intelligence/) |
| U-Net | Ronneberger et al.（Freiburg） | 编码-解码 + 跳跃连接，医学图像分割的经典架构 | [01·视觉线](../01-Visual-Intelligence/) |
| DCGAN | Radford et al.（OpenAI） | 卷积 GAN + 训练稳定技巧，生成真实图像的第一个可复现方法 | [01·视觉线](../01-Visual-Intelligence/) |
| Neural Style Transfer | Gatys et al.（Tübingen） | 内容 / 风格分离迁移，生成式 AI 第一次破圈进入大众视野 | [01·视觉线](../01-Visual-Intelligence/) |
| Deep Speech 2 | Amodei et al.（Baidu） | 端到端语音识别，超越人类水平，大数据 + 大模型的工业验证 | [01·视觉线](../01-Visual-Intelligence/) |
| Spatial Transformer | Jaderberg et al.（DeepMind） | 可学习空间变换，让网络具备几何注意力 | [01·视觉线](../01-Visual-Intelligence/) |
| YOLO v1 | Redmon et al. | 单阶段实时目标检测，效率革命，工业落地新选择 | [01·视觉线](../01-Visual-Intelligence/) |
| char-rnn / 语言生成 | Karpathy | LSTM 字符级语言模型，展示 RNN 惊人的文本生成能力 | [02·语言线](../02-Language-Transformers/) |

---

<a id="2016"></a>
## 2016 · AlphaGo：强化学习登上历史舞台

### 之前的世界

围棋是当时 AI 最后一个没有攻克的传统棋类游戏。它的搜索空间比国际象棋大出天文数字，暴力搜索完全不可能。专家们普遍认为 AI 至少还需要十年才能挑战顶尖人类棋手。

### 发生了什么

2016 年 3 月，DeepMind 的 AlphaGo 以 4:1 击败世界冠军李世石。

AlphaGo 的核心是三个技术的结合：
- **深度卷积网络**：评估当前棋局的胜率（价值网络）和推荐落子位置（策略网络）
- **蒙特卡洛树搜索**：有限地向前搜索
- **强化学习**：让两个网络自我对弈，不断强化好的落子策略

这场比赛的意义不只是围棋。它向世界展示了：**强化学习 + 深度网络，可以在极其复杂的决策问题上超越人类**。这一思路后来被直接用于对齐大模型（RLHF）。

### 解决了什么，又带来了什么新问题

AlphaGo 证明了强化学习的实用性，但也暴露了它的局限：训练需要**海量的自我对弈**，样本效率极低。如何让模型用更少的经验学到更多，成为 RL 领域接下来几年的核心问题。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| WaveNet | van den Oord et al.（DeepMind） | 自回归波形生成，语音合成质量飞跃，影响后续音频生成模型 | [01·视觉线](../01-Visual-Intelligence/) |
| FastText | Joulin et al.（Facebook） | 子词 n-gram 表示，词向量训练速度极快，适合小语种 | [02·语言线](../02-Language-Transformers/) |
| DenseNet | Huang et al.（Cornell） | 每层连接所有前层，特征复用最大化，参数效率极高 | [01·视觉线](../01-Visual-Intelligence/) |
| NAS（神经架构搜索） | Zoph & Le（Google Brain） | 用 RL 搜索网络结构，AutoML 和 LLM 架构设计方向起点 | [03·规模多模态](../03-Scale-Multimodal/) |
| A3C | Mnih et al.（DeepMind） | 异步并行 Actor-Critic，RL 训练效率突破，RLHF 基础技术 | [00·前置](../00-Prerequisites/) |
| OpenAI Gym | Brockman et al.（OpenAI） | RL 环境标准化接口，让强化学习研究可复现、可对比 | [00·前置](../00-Prerequisites/) |
| SqueezeNet | Iandola et al.（Berkeley） | 50× 小于 AlexNet 且精度相当，轻量化模型的早期代表 | [01·视觉线](../01-Visual-Intelligence/) |
| Pointer Networks | Vinyals et al.（Google） | 注意力直接指向输入位置，解决变长结构化输出问题 | [02·语言线](../02-Language-Transformers/) |
| SSD 目标检测 | Liu et al. | 多尺度特征图实时检测，精度-速度平衡的新基准 | [01·视觉线](../01-Visual-Intelligence/) |
| OpenAI 成立 | OpenAI | 非营利 AI 安全研究机构成立，GPT 系列的机构起点 | [03·规模多模态](../03-Scale-Multimodal/) |

---

<a id="2017"></a>
## 2017 · Transformer：把 RNN 彻底扔掉

### 之前的世界

做序列任务（翻译、语言建模），大家用的是 LSTM + Attention。这套组合有效，但有一个无法绕过的缺点：**RNN 必须按时间步依次计算，天生无法并行**。句子越长，训练越慢。

### 发生了什么

Google 的 Vaswani 等人发表了《Attention Is All You Need》，提出 **Transformer 架构**。

核心判断只有一句话：**RNN 是多余的，Attention 本身就足够了**。

Transformer 用**自注意力（Self-Attention）**替代了 RNN，让序列中的每个位置都可以直接关注其他任意位置，不再需要顺序传递信息。这带来了两个根本性的改变：

1. **完全并行**：整个序列同时计算，GPU 的并行能力被充分利用
2. **长距离依赖不衰减**：位置 1 和位置 100 的关系，和位置 1 与位置 2 的关系一样直接

几个关键设计：
- **多头注意力**：用多个注意力头同时关注不同类型的关系（语法、语义、位置……）
- **位置编码**：因为去掉了 RNN，需要额外告诉模型词的顺序
- **残差连接 + Layer Norm**：借鉴 ResNet，保证深层网络可以训练

### 解决了什么，又带来了什么新问题

Transformer 解放了训练速度，让"用更大的模型、更多的数据"成为可能。

新问题：自注意力的复杂度是 **O(n²)**——序列长度翻倍，计算量翻四倍。处理长文本时，显存直接爆掉。这个问题贯穿了之后好几年的研究。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| AlphaGo Zero | Silver et al.（DeepMind） | 纯自我对弈无人类棋谱，RL 自举能力的极致证明 | [00·前置](../00-Prerequisites/) |
| PPO | Schulman et al.（OpenAI） | 近端策略优化，RLHF 的核心算法，OpenAI 主力 RL 工具 | [00·前置](../00-Prerequisites/) |
| MobileNet | Howard et al.（Google） | 深度可分离卷积，移动端深度学习效率基准 | [01·视觉线](../01-Visual-Intelligence/) |
| SE-Net | Hu et al. | 通道注意力即插即用模块，ILSVRC 2017 冠军 | [01·视觉线](../01-Visual-Intelligence/) |
| Focal Loss / RetinaNet | Lin et al.（Facebook） | 解决类别不平衡，单阶段检测性能追上两阶段 | [01·视觉线](../01-Visual-Intelligence/) |
| Capsule Networks | Hinton et al.（Google） | 动态路由替代池化，挑战 CNN 空间表示方式 | [01·视觉线](../01-Visual-Intelligence/) |
| Progressive GAN | Karras et al.（NVIDIA） | 渐进式分辨率生成，高质量人脸图像的里程碑 | [01·视觉线](../01-Visual-Intelligence/) |
| PyTorch 主流化 | Paszke et al.（Facebook） | 动态图框架成为研究界标准，影响深度学习生态 10 年 | [00·前置](../00-Prerequisites/) |
| Soft Actor-Critic | Haarnoja et al.（Berkeley） | 最大熵 RL，连续控制任务高样本效率基准 | [00·前置](../00-Prerequisites/) |
| 多头注意力理论分析 | — | Transformer 各头关注不同语言结构的实证研究开始 | [02·语言线](../02-Language-Transformers/) |

---

<a id="2018"></a>
## 2018 · BERT + GPT-1：预训练时代正式开启

### 之前的世界

Word2Vec 的词向量是静态的，而且每个词只有一个表示。用这些向量做下游任务，还是需要在每个任务上从头训练大量参数。

ELMo（Allen AI，2018 年初）迈出了一步：用双向 LSTM 生成上下文相关的词向量。但 LSTM 本身的局限限制了它的上限。

### 发生了什么

2018 年是"预训练 + 微调"范式确立的一年，两个模型分别代表两条路线：

**GPT-1（OpenAI）**：单向（从左到右）Transformer，在大量文本上做语言模型预训练，然后在下游任务上微调。思路简单，但验证了 Transformer 预训练的可行性。

**BERT（Google）**：双向 Transformer，用两个训练目标：
- **Masked Language Model（MLM）**：随机遮住 15% 的词，让模型猜被遮住的是什么
- **Next Sentence Prediction（NSP）**：判断两句话是否相邻

BERT 在 11 个 NLP 任务上同时刷新了最好成绩，震动了整个 NLP 社区。

### 解决了什么，又带来了什么新问题

**一词多义**问题终于被解决——同一个词在不同句子里会得到不同的向量。

**"预训练 + 微调"**取代了"每个任务从头训练"，成为 NLP 的新范式。

新问题：微调仍然需要每个任务有标注数据。能不能连微调都省掉？这是 GPT 路线接下来要回答的问题。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| ELMo | Peters et al.（AllenAI） | 双向 LSTM 上下文词向量，动态表示一词多义，BERT 前身 | [02·语言线](../02-Language-Transformers/) |
| ULMFiT | Howard & Ruder（Fast.ai） | 语言模型预训练+判别微调，NLP 迁移学习的最早系统验证 | [02·语言线](../02-Language-Transformers/) |
| GLUE Benchmark | Wang et al.（NYU 等） | 统一 9 个 NLP 任务评测，推动预训练模型进入 NLP 主流 | [02·语言线](../02-Language-Transformers/) |
| Transformer-XL | Dai et al.（Google/CMU） | 片段级循环机制，突破固定上下文长度，XLNet 的直接前身 | [02·语言线](../02-Language-Transformers/) |
| OpenAI Five（Dota 2） | OpenAI | RL 在高复杂度多人游戏中战胜职业选手，大规模 RL 系统验证 | [00·前置](../00-Prerequisites/) |
| BigGAN | Brock et al.（DeepMind） | 大批量 GAN 训练，生成图像质量和多样性大幅提升 | [01·视觉线](../01-Visual-Intelligence/) |
| PyTorch 1.0 正式版 | Paszke et al.（Facebook） | 生产级动态图框架发布，研究-工业统一，TensorFlow 的主要竞争者 | [00·前置](../00-Prerequisites/) |
| SNAIL | Mishra et al.（Berkeley） | 时序卷积 + 注意力元学习，few-shot 学习新思路 | [00·前置](../00-Prerequisites/) |
| XLNet 研究启动 | Yang et al.（CMU/Google） | 排列语言模型思路形成，下一年正式超越 BERT | [02·语言线](../02-Language-Transformers/) |
| BERT 中文版 | Google | BERT 多语言支持，NLP 预训练向全球语言扩展 | [02·语言线](../02-Language-Transformers/) |

---

<a id="2019"></a>
## 2019 · GPT-2 + T5：规模的野心开始显现

### 之前的世界

BERT 统治了 NLP 排行榜，大家都在"BERT + 微调"这条路上内卷。但 OpenAI 在想另一件事：**语言模型本身，能不能就是通用 AI 的雏形？**

### 发生了什么

**GPT-2（OpenAI，2019 年）**：1.5B 参数，在 Reddit 高质量帖子（WebText）上训练。生成的文本质量高到 OpenAI 起初不敢开源——他们发了一篇博文说"这个模型太危险了，我们先只发小版本"。

虽然这个决定后来被广泛嘲讽（实际危险程度被高估了），但 GPT-2 的文本生成能力确实让人眼前一亮：连贯、有逻辑、风格统一。

**T5（Google，2019 年）**："Text-to-Text Transfer Transformer"——把所有 NLP 任务统一成一种格式：**输入一段文字，输出一段文字**。翻译、摘要、问答、分类，全都变成同一个框架。这是"大一统"思路的早期体现。

同年，**RoBERTa（Facebook）**证明 BERT 其实训练不够充分：去掉 NSP、增大 batch size、训练更长时间，效果大幅提升。

### 解决了什么，又带来了什么新问题

GPT-2 展示了语言模型规模化的潜力，T5 提出了任务统一的思路。

新问题：1.5B 参数已经很大了，但似乎还不够。**更大的模型会出现质变吗？** OpenAI 开始思考这个问题，答案要等到 2020 年。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| XLNet | Yang et al.（CMU/Google） | 排列语言模型，同时获取双向上下文，全面超越 BERT | [02·语言线](../02-Language-Transformers/) |
| ALBERT | Lan et al.（Google） | 参数共享 + 因子化嵌入，18× 小于 BERT-large 但性能不降 | [02·语言线](../02-Language-Transformers/) |
| DistilBERT | Sanh et al.（HuggingFace） | 知识蒸馏压缩 BERT 到 60%，推理快 60%，轻量化 NLP 起点 | [02·语言线](../02-Language-Transformers/) |
| Sparse Transformer | Child et al.（OpenAI） | 稀疏注意力模式，把长序列自注意力从 O(n²) 往下打 | [02·语言线](../02-Language-Transformers/) |
| Megatron-LM | Shoeybi et al.（NVIDIA） | 模型并行训练框架，8.3B 参数，大规模分布式训练基础设施 | [05·系统生产](../05-Systems-Production/) |
| HuggingFace Transformers | Wolf et al.（HuggingFace） | 统一预训练模型生态，成为开源 NLP 的事实标准平台 | [02·语言线](../02-Language-Transformers/) |
| ERNIE（百度） | Sun et al.（Baidu） | 知识增强预训练，把知识图谱融入 BERT，中文 NLP 里程碑 | [02·语言线](../02-Language-Transformers/) |
| MoCo | He et al.（Facebook） | 动量对比学习，自监督视觉表示学习可行性验证 | [00·前置](../00-Prerequisites/) |
| GPT-2 全量开源 | OpenAI | 最终开源 1.5B 版本，验证开源大模型对社区的推动价值 | [03·规模多模态](../03-Scale-Multimodal/) |
| SuperGLUE | Wang et al. | 比 GLUE 更难的 NLP 评测集，推动模型突破人类基准 | [02·语言线](../02-Language-Transformers/) |

---

<a id="2020"></a>
## 2020 · GPT-3 + Scaling Laws：大力出奇迹

### 之前的世界

研究者们普遍认为，大模型需要在每个任务上微调，才能达到好的效果。没有人知道，单纯把模型做大，会发生什么。

### 发生了什么

**Scaling Laws（OpenAI，2020 年 1 月）**：Kaplan 等人发现，模型性能和三个因素之间存在**可预测的幂律关系**：模型参数量、训练数据量、计算量（FLOPs）。这意味着：**在花钱训练之前，就可以预测训练结果**。

这篇论文的意义极其深远——它把"大模型研究"从玄学变成了工程。

**GPT-3（OpenAI，2020 年 5 月）**：175B 参数，比 GPT-2 大 100 倍。

但真正令人震惊的不是参数量，而是一种新能力的涌现：**Few-shot Learning（少样本学习）**。

你不需要微调，只需要在 prompt 里给几个例子，GPT-3 就能举一反三。这叫 **In-Context Learning（上下文学习）**——模型从 prompt 里"临时学习"，而不需要更新参数。

### 解决了什么，又带来了什么新问题

GPT-3 证明了 **Scale 本身是一种能力**。足够大的模型，无需微调就能解决很多任务。

新问题出现了三个方向：
1. **对齐问题**：GPT-3 什么都会说，包括有害、不真实的内容。怎么让它"听话"？
2. **成本问题**：训练一次 GPT-3 要花几百万美元，只有极少数机构能做
3. **幻觉问题**：模型会自信地说错话，而且你很难区分它什么时候在胡说

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| Vision Transformer（ViT） | Dosovitskiy et al.（Google） | 纯 Transformer 处理图像 patch，打破 CNN 对视觉的垄断 | [03·规模多模态](../03-Scale-Multimodal/) |
| AlphaFold 2 | Jumper et al.（DeepMind） | 蛋白质结构预测媲美实验精度，AI for Science 里程碑 | [00·前置](../00-Prerequisites/) |
| RAG 论文 | Lewis et al.（Facebook） | 检索 + 生成结合，缓解幻觉 + 知识截止问题，RAG 方向正式提出 | [05·系统生产](../05-Systems-Production/) |
| Switch Transformer | Fedus et al.（Google） | 混合专家 MoE，相同算力参数量突破万亿，MoE 主流化前身 | [03·规模多模态](../03-Scale-Multimodal/) |
| SimCLR | Chen et al.（Google） | 对比学习自监督预训练，视觉表示无标注学习成熟 | [00·前置](../00-Prerequisites/) |
| Big Bird | Zaheer et al.（Google） | 稀疏全局注意力，4096 tokens 长文档理解 Transformer | [02·语言线](../02-Language-Transformers/) |
| ELECTRA | Clark et al.（Google/Stanford） | 替换词检测训练目标，更高效的 BERT 替代预训练方案 | [02·语言线](../02-Language-Transformers/) |
| DALL-E 研发开始 | OpenAI | 文本 → 图像生成的技术积累，多模态 LLM 的蓄水期 | [03·规模多模态](../03-Scale-Multimodal/) |
| GPT-3 API 公测 | OpenAI | LLM 以 API 形式提供服务，商业化 AI 应用生态起点 | [05·系统生产](../05-Systems-Production/) |
| Chinchilla 前置研究 | DeepMind | 训练效率与数据量关系的早期研究，奠定 2022 年 Chinchilla 论文 | [03·规模多模态](../03-Scale-Multimodal/) |

---

<a id="2021"></a>
## 2021 · CLIP + Codex + LoRA：多模态与效率的破局

### 之前的世界

视觉模型只认图，语言模型只认字，两个世界老死不相往来。代码生成靠规则和模板，不够灵活。大模型微调成本高昂，小机构根本负担不起。

### 发生了什么

**CLIP（OpenAI）**：用 4 亿对图文数据，同时训练一个图像编码器和一个文本编码器，用**对比学习**让匹配的图文对在向量空间里靠近。训练完之后，CLIP 可以做零样本图像分类——给它一张图和一些文字描述，它能判断哪个描述最匹配，无需在目标数据集上训练过。

这是**多模态理解**的一个关键里程碑，也是后来 GPT-4V、DALL-E 等模型的基础。

**Codex（OpenAI）**：在 GitHub 代码上微调的 GPT，驱动了 **GitHub Copilot**。程序员第一次有了真正可用的 AI 结对编程工具。代码补全、函数生成、注释转代码，全部变成现实。

**LoRA（Hu 等人）**：大模型微调一直面临一个问题：175B 的模型，微调要存多少梯度？LoRA 的思路是，不动原始权重，只在旁边接两个低秩矩阵，让它们学任务相关的变化。参数量可以减少到原来的 **1/1000**，效果几乎不打折扣。

这一技术让"个人也能微调大模型"成为可能，开源生态的爆发从这里埋下了种子。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| DALL-E | Ramesh et al.（OpenAI） | 首个高质量文本生图模型，多模态生成进入公众视野 | [03·规模多模态](../03-Scale-Multimodal/) |
| AlphaFold 2 发表 | Jumper et al.（DeepMind） | Nature 正式发表，公开 2 亿蛋白质结构预测数据库 | [00·前置](../00-Prerequisites/) |
| FLAN（指令微调） | Wei et al.（Google） | 多任务指令数据微调 LLM，zero-shot 能力大幅提升，ChatGPT 前身 | [03·规模多模态](../03-Scale-Multimodal/) |
| InstructGPT 研发 | Ouyang et al.（OpenAI） | RLHF 对齐技术落地，GPT-3 向"有用、无害、诚实"对齐 | [03·规模多模态](../03-Scale-Multimodal/) |
| WebGPT | Nakano et al.（OpenAI） | 让语言模型使用搜索引擎，早期 Tool Use / RAG 雏形 | [05·系统生产](../05-Systems-Production/) |
| Decision Transformer | Chen et al.（Berkeley/Google） | RL 问题变序列预测，Transformer 解决决策问题的新范式 | [00·前置](../00-Prerequisites/) |
| Gopher | Rae et al.（DeepMind） | 280B 参数语言模型，DeepMind 进入大模型军备竞赛 | [03·规模多模态](../03-Scale-Multimodal/) |
| Perceiver IO | Jaegle et al.（DeepMind） | 通用架构处理任意模态，多模态统一表示的早期探索 | [03·规模多模态](../03-Scale-Multimodal/) |
| GitHub Copilot 公测 | GitHub / OpenAI | Codex 驱动的 AI 编程助手，开发者工具 AI 化的起点 | [05·系统生产](../05-Systems-Production/) |
| Megatron-LM v2 | NVIDIA | 支持千亿参数模型训练，大规模并行训练基础设施成熟 | [05·系统生产](../05-Systems-Production/) |

---

<a id="2022"></a>
## 2022 · ChatGPT + RLHF：AI 第一次真正走进大众

### 之前的世界

GPT-3 很强，但用起来别扭——它是一个"文本补全"模型，不是"对话"模型。你问它问题，它可能继续补全问题，而不是回答。更大的问题是，它不在乎你的感受，有害内容、偏见、谎言，它照单全说。

### 发生了什么

**InstructGPT / ChatGPT（OpenAI）**的核心技术是 **RLHF（基于人类反馈的强化学习）**，分三步：

1. **监督微调（SFT）**：用人类示范的"好回答"来微调模型
2. **奖励模型训练（RM）**：让人类对不同回答排名，训练一个"评分模型"
3. **PPO 强化学习**：用奖励模型的分数作为信号，继续优化语言模型

这套流程让模型从"预测下一个词"变成了"学习怎么让人满意"。

2022 年 11 月 30 日，ChatGPT 上线。5 天后，用户破百万。两个月后，突破 1 亿用户，成为**史上增长最快的消费应用**。

同年，**Stable Diffusion（Stability AI）**开源了文字生成图像模型，**Midjourney** 进入公测，AI 绘画爆炸式普及。**Chinchilla（DeepMind）**则证明了此前的大模型都"训练不足"——用同样的计算量，更小的模型 + 更多数据，效果更好。

### 解决了什么，又带来了什么新问题

RLHF 初步解决了**对齐问题**：模型变得更有用、更无害、更诚实。

但新的问题随之而来：
- **过度对齐**（Alignment Tax）：模型变得过于谨慎，拒绝很多本无害的问题
- **奖励 Hacking**：模型学会"讨好"奖励模型，而不是真正有帮助
- **幻觉依旧存在**：对齐不能解决模型"自信说错话"的问题

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| PaLM | Chowdhery et al.（Google） | 540B 参数，Chain-of-Thought 推理能力首次系统研究 | [03·规模多模态](../03-Scale-Multimodal/) |
| DALL-E 2 | Ramesh et al.（OpenAI） | CLIP + 扩散模型，图像质量和可控性大幅提升 | [03·规模多模态](../03-Scale-Multimodal/) |
| Whisper | Radford et al.（OpenAI） | 大规模弱监督语音识别，多语言 ASR 开源解决方案 | [03·规模多模态](../03-Scale-Multimodal/) |
| Constitutional AI | Bai et al.（Anthropic） | 用 AI 反馈训练 AI，减少人工标注，对齐方法的重要变体 | [03·规模多模态](../03-Scale-Multimodal/) |
| DPO | Rafailov et al.（Stanford） | 直接偏好优化，不需要独立奖励模型，比 RLHF 更稳定简洁 | [03·规模多模态](../03-Scale-Multimodal/) |
| Flamingo | Alayrac et al.（DeepMind） | 少样本视觉-语言模型，多模态 In-Context Learning 先驱 | [03·规模多模态](../03-Scale-Multimodal/) |
| OPT | Zhang et al.（Meta） | 175B 参数开源语言模型，推动学术界大模型研究民主化 | [03·规模多模态](../03-Scale-Multimodal/) |
| Minerva | Lewkowycz et al.（Google） | 数学推理专用 LLM，STEM 领域 LLM 应用探索 | [03·规模多模态](../03-Scale-Multimodal/) |
| Flash Attention | Dao et al.（Stanford） | IO 感知注意力计算，训练速度 2-4×，显存节省 5-20× | [05·系统生产](../05-Systems-Production/) |
| LLaMA 研发准备 | Meta | 效率优先路线启动，更小模型 + 更多数据的技术路线 | [03·规模多模态](../03-Scale-Multimodal/) |

---

<a id="2023"></a>
## 2023 · GPT-4 + LLaMA：开源的反击

### 之前的世界

大模型基本是 OpenAI、Google、Anthropic 等少数公司的专利。研究者没有开放的权重，无法复现、改进、研究这些模型的内部机制。

### 发生了什么

**GPT-4（OpenAI，2023 年 3 月）**：多模态（能看图），推理能力大幅提升，通过了律师资格考试前 10%、GRE 各科高分。参数量官方未公布，但表现质的飞跃。

**LLaMA（Meta，2023 年 2 月）**：Meta 开放了 LLaMA 的模型权重（最初通过学术申请，后来直接开源）。参数从 7B 到 65B，在同等规模下性能逼近 GPT-3.5。

LLaMA 的开源直接点燃了社区：

- **Alpaca**：Stanford 用 52K 条指令数据微调 LLaMA，一个周末，$600
- **Vicuna**：用 ChatGPT 对话数据微调，接近 ChatGPT 水平
- **LLaMA 2（Meta，2023 年 7 月）**：正式开源可商用，生态全面爆发
- **Mistral 7B**：7B 参数打赢 13B 模型，效率极致

同年，**RAG（检索增强生成）**成为主流解决方案：给模型配一个外部知识库，先检索相关文档，再生成回答，解决"知识截止日期"和"幻觉"问题。

**DPO（Direct Preference Optimization）**作为 RLHF 的简化替代出现，不需要单独训练奖励模型，直接优化偏好，训练更稳定。

### 解决了什么，又带来了什么新问题

LLaMA 打破了大模型的垄断格局，开源社区和闭源公司开始真正竞争。

新问题：**Agent（智能体）**的潮流涌现——AutoGPT 等项目爆红，大家开始探索让模型自主使用工具、完成多步任务。但当时的模型工具调用能力不稳定，Agent 更多还是 Demo 阶段。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| LLaMA 2 | Touvron et al.（Meta） | 正式开源可商用，7B/13B/70B，开源 LLM 生态全面爆发 | [03·规模多模态](../03-Scale-Multimodal/) |
| Mistral 7B | Jiang et al.（Mistral AI） | 滑动窗口注意力 + GQA，7B 打赢 13B，效率极致 | [03·规模多模态](../03-Scale-Multimodal/) |
| Claude | Anthropic | Constitutional AI 对齐，100K 长上下文，商业闭源竞争 | [03·规模多模态](../03-Scale-Multimodal/) |
| Code Llama | Rozière et al.（Meta） | 代码专用大模型，开源代码生成的新基准 | [03·规模多模态](../03-Scale-Multimodal/) |
| ReAct / Chain-of-Thought | Yao et al. | 推理 + 行动循环框架，Agent 工程化的理论基础 | [05·系统生产](../05-Systems-Production/) |
| AutoGPT | Toran Bruce Richards | 让 LLM 自主分解并执行多步任务，Agent 破圈进入大众视野 | [05·系统生产](../05-Systems-Production/) |
| LangChain / LlamaIndex | 开源社区 | RAG 与 Agent 工程化框架，LLM 应用开发基础设施成型 | [05·系统生产](../05-Systems-Production/) |
| Mixtral 8x7B | Mistral AI | MoE 开源，46B 参数只激活 13B，高效率高质量兼顾 | [03·规模多模态](../03-Scale-Multimodal/) |
| Llama 2 Chat | Meta | RLHF 对齐版 Llama 2，开源对话模型的新基准 | [03·规模多模态](../03-Scale-Multimodal/) |
| GPT-4 Technical Report | OpenAI | 详细披露训练流程与能力评测，推动整个行业对标 | [03·规模多模态](../03-Scale-Multimodal/) |

---

<a id="2024"></a>
## 2024 · MoE + 长上下文 + 推理模型：效率与能力的双向突破

### 之前的世界

大模型在变强，但训练和推理的成本也在线性增加。同时，模型对"复杂推理"的处理仍然不尽人意——遇到数学、代码里的逻辑题，经常一步走错满盘皆输。

### 发生了什么

**Mixtral 8x7B（Mistral AI，2024 年）**：**混合专家模型（MoE）**进入主流。思路是：把模型拆成多个"专家"子网络，每次推理只激活其中几个。参数总量 46B，但推理时只用约 13B，效果接近 70B 的稠密模型，速度却快得多。

**长上下文**成为各家竞相突破的方向：Gemini 1.5 Pro 支持 **100 万 Token** 的上下文窗口，相当于一次处理几十本书。长上下文解决了 RAG 的一部分问题，但也带来了"大海捞针"时注意力分散的挑战。

**o1（OpenAI，2024 年 9 月）**：最重要的范式转变之一。o1 在推理时会先做"慢思考"——生成一段内部 Chain-of-Thought，再给出最终答案。训练时用强化学习奖励正确的推理过程，而不只是正确答案。

结果是：在数学竞赛、代码、科学题上，o1 的表现大幅超越之前的模型，接近博士生水平。

**Llama 3、Qwen 2、Gemma** 等开源模型纷纷跟进，7B/8B 级别的小模型性能接近一年前的 GPT-3.5。

### 解决了什么，又带来了什么新问题

MoE 解决了"大模型推理太贵"的问题，长上下文解决了"装不下整个文档"的问题，o1 开辟了"用推理时间换准确率"的新方向。

新问题：**推理成本**随之上升（o1 的每次调用比 GPT-4 贵数倍），如何高效地分配"思考预算"成为新的研究方向。

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| Claude 3 系列 | Anthropic | Haiku/Sonnet/Opus 三档，长上下文 + 多模态，对齐能力领先 | [03·规模多模态](../03-Scale-Multimodal/) |
| GPT-4o | OpenAI | 统一多模态（文本/图像/语音），实时交互，端到端训练 | [03·规模多模态](../03-Scale-Multimodal/) |
| Sora | Liu et al.（OpenAI） | 文本生成高质量长视频，世界模型雏形，视频生成里程碑 | [03·规模多模态](../03-Scale-Multimodal/) |
| Llama 3 | Meta | 8B/70B 开源，代码与推理能力大幅提升，开源新基准 | [03·规模多模态](../03-Scale-Multimodal/) |
| DeepSeek V2/V3 | DeepSeek | MoE 架构高效开源，性价比极高，挑战闭源头部模型 | [03·规模多模态](../03-Scale-Multimodal/) |
| Qwen 2 系列 | Alibaba | 全系列开源，多语言，中文 LLM 开源生态成熟标志 | [03·规模多模态](../03-Scale-Multimodal/) |
| Gemma | Google | 小型开源模型，研究友好，支持边缘设备部署 | [03·规模多模态](../03-Scale-Multimodal/) |
| vLLM / PagedAttention | Kwon et al.（Berkeley） | LLM 推理服务效率革命，吞吐量提升 24×，成为服务基础设施 | [05·系统生产](../05-Systems-Production/) |
| Flash Attention 2/3 | Dao et al. | 进一步提升注意力计算效率，支持更长上下文训练 | [05·系统生产](../05-Systems-Production/) |
| RLVR（可验证强化学习） | — | 用可验证奖励（数学/代码结果）训练推理，o1 技术路线拆解 | [03·规模多模态](../03-Scale-Multimodal/) |

---

<a id="2025"></a>
## 2025 · 推理模型 + DeepSeek：开源追平，范式再变

### 之前的世界

推理型模型（o1 系列）还是 OpenAI 的独门武器，开源世界在这个方向几乎空白。

### 发生了什么

**DeepSeek R1（DeepSeek，2025 年 1 月）**：中国创业公司 DeepSeek 发布了开源推理模型，在数学、代码等推理任务上对标 o1，同时开放权重和技术报告。

技术亮点：
- 用**纯强化学习**训练出推理能力，不依赖大量人工标注的推理轨迹
- 训练成本远低于 OpenAI 的估算，挑战了"只有砸钱才能做大模型"的认知
- 完整的技术报告引发全球研究者的分析热潮

**测试时计算扩展（Test-Time Compute Scaling）**成为 2025 年的核心话题：与其训练一个更大的模型，不如让现有模型在推理时"多想一会儿"。这个方向暗示，**AI 能力的提升不再只依赖更多训练数据**。

多模态 Agent 开始真正可用：视觉 + 语言 + 工具调用逐渐成熟，模型可以"看着屏幕操作电脑"。

### 解决了什么，又带来了什么新问题

开源世界终于在推理能力上追平了顶尖闭源模型，民主化的趋势进一步加速。

新的问题和方向正在形成：
- **长链推理的可靠性**：模型会"想太多"走弯路，甚至在错误的前提上越想越深
- **Agent 的可靠性**：真实世界的任务需要数十步操作，任何一步出错可能导致全盘失败
- **评估体系的失效**：传统 benchmark 开始"过拟合"，真实能力难以衡量

### 同年关键工作速览

| 工作 | 提出者 / 机构 | 核心贡献 | 所属模块 |
|------|--------------|----------|----------|
| o3 / o4-mini | OpenAI | 推理能力进一步提升，ARC-AGI 接近人类，推理模型天花板 | [03·规模多模态](../03-Scale-Multimodal/) |
| Claude 3.5 / 3.7 | Anthropic | 扩展思考（Extended Thinking），Hybrid 推理模式 | [03·规模多模态](../03-Scale-Multimodal/) |
| Llama 4 | Meta | MoE 架构 + 原生多模态，开源旗舰新突破 | [03·规模多模态](../03-Scale-Multimodal/) |
| Gemini 2.0 | Google | 原生多模态 + Agent 能力，2M tokens 长上下文 | [03·规模多模态](../03-Scale-Multimodal/) |
| Test-Time Compute Scaling | — | 推理时计算扩展规律，与训练时 Scaling Laws 并列为 AI 两大定律 | [03·规模多模态](../03-Scale-Multimodal/) |
| 多模态 Agent 成熟化 | — | 视觉 + 语言 + 工具调用统一，计算机操作 Agent 进入可用阶段 | [05·系统生产](../05-Systems-Production/) |
| Benchmark 失效危机 | — | MMLU/HumanEval 等主流评测集接近饱和，新一代评估体系重建 | [05·系统生产](../05-Systems-Production/) |
| Speculative Decoding 普及 | — | 草稿模型 + 验证模型并行，LLM 推理速度 2-3× 提升 | [05·系统生产](../05-Systems-Production/) |
| 推理 Budget 控制 | — | 分配"思考预算"，避免模型过度推理，推理效率新方向 | [03·规模多模态](../03-Scale-Multimodal/) |
| 长链推理可靠性研究 | — | 错误前提上的深度推理放大偏差，AI 安全与对齐新挑战 | [03·规模多模态](../03-Scale-Multimodal/) |

---

## 如何阅读这条时间线

每一年的突破，都可以用三个问题来理解：

| 问题 | 含义 |
|------|------|
| **之前卡在哪** | 旧方法的天花板是什么 |
| **谁、用什么方法突破了** | 核心技术思路 |
| **解决了什么，留下了什么** | 进步的同时产生了哪些新问题 |

技术不是凭空出现的，它是被前一代技术的缺陷**逼出来的**。

各章节详细展开 → 见对应模块目录下的 README.md 与 notebook.ipynb

| 模块 | 覆盖的时间线节点 |
|------|----------------|
| [00·前置准备](../00-Prerequisites/) | 神经网络基础 |
| [01·视觉线](../01-Visual-Intelligence/) | 2012–2017 CNN · RNN · 生成模型 |
| [02·语言线](../02-Language-Transformers/) | 2013–2019 词向量 · Attention · BERT |
| [03·规模与多模态](../03-Scale-Multimodal/) | 2020–2021 GPT-3 · ViT · CLIP |
| [04·对齐与开源](../04-Alignment-OpenSource/) | 2022–2023 RLHF · DPO · LLaMA |
| [05·系统与生产](../05-Systems-Production/) | 2023–2025 RAG · Agent · 推理 · MLOps |
| [06·实战项目](../06-Capstone-Projects/) | 跨阶段综合 |
