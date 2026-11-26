# 从零学机器学习、深度学习与大模型：完整大纲

> 由 [roadmap.json](roadmap.json) 生成（`node learn/build-roadmap.mjs`）。6 层、45 个知识块、191 个知识点，已完成 17。
> 每个知识块的旧版素材位置记录在 roadmap.json 的 `old` 字段（`_archive/v1/` 下，质量未重新核对，写新版时需重写）。

![](../.github/assets/roadmap.png)

### 0 · 数学与工具

读得懂公式，写得出代码。共 7 个知识块、32 个知识点，已完成 0。

| 知识块 | 包含的知识点 | 状态 |
|---|---|---|
| **线性代数** | 向量与矩阵运算、线性变换与几何直觉、特征值与特征分解、SVD 与低秩近似、范数、内积与相似度 | 待写 |
| **微积分与梯度** | 导数与链式法则、偏导、梯度、Jacobian、Hessian、泰勒展开、自动微分原理 | 待写 |
| **概率与统计** | 随机变量与常见分布、期望、方差、协方差、贝叶斯公式、最大似然与 MAP、置信区间与假设检验、大数定律与中心极限定理 | 待写 |
| **信息论** | 熵、交叉熵、KL 散度、互信息、困惑度 | 待写 |
| **优化基础** | 凸与非凸、梯度下降、牛顿法、拉格朗日与约束优化、随机优化直觉 | 待写 |
| **Python 科学计算** | NumPy 向量化、Matplotlib 画图、Jupyter 工作流 | 待写 |
| **PyTorch 入门** | 张量与广播、autograd、nn.Module 与训练循环、GPU 与调试 | 待写 |

### 1 · 机器学习

学会“学习”这件事。共 7 个知识块、29 个知识点，已完成 0。

| 知识块 | 包含的知识点 | 状态 |
|---|---|---|
| **学习问题与评估** | 监督 / 无监督 / 强化学习、偏差与方差、过拟合与正则化、交叉验证与数据划分、分类与回归指标（精召 F1 AUC）、数据泄漏与类别不均衡、特征工程 | 待写 |
| **线性模型** | 线性回归、逻辑回归、Softmax 回归、L1 / L2 正则 | 待写 |
| **树与集成** | 决策树、Bagging 与随机森林、AdaBoost、GBDT / XGBoost | 待写 |
| **核方法** | SVM 与间隔、核技巧 | 待写 |
| **概率模型** | 朴素贝叶斯、高斯混合与 EM、隐马尔可夫模型、概率图模型入门 | 待写 |
| **无监督学习** | K-means、层次聚类与 DBSCAN、PCA、t-SNE 与 UMAP | 待写 |
| **强化学习基础** | MDP 与价值函数、Q-learning、策略梯度、Actor-Critic 与 PPO | 待写 |

### 2 · 深度学习

神经网络怎么训练、怎么设计。共 9 个知识块、31 个知识点，已完成 0。

| 知识块 | 包含的知识点 | 状态 |
|---|---|---|
| **神经网络与反向传播** | 感知机与 MLP、反向传播推导、万能近似与深度的意义 | 待写 |
| **激活、初始化与损失** | ReLU / GELU / SwiGLU、Xavier 与 He 初始化、MSE、交叉熵、对比损失 | 待写 |
| **优化器与学习率** | SGD 与动量、Adam / AdamW、学习率调度与 warmup、梯度裁剪 | 待写 |
| **归一化与正则化** | BatchNorm / LayerNorm / RMSNorm、Dropout 与权重衰减、早停与数据增强 | 待写 |
| **训练实践** | 混合精度、梯度消失与爆炸排查、过拟合小批量测试、实验记录与复现 | 待写 |
| **卷积网络** | 卷积与池化、LeNet / AlexNet / VGG、Inception 与 ResNet、EfficientNet 与 ConvNeXt | 待写 |
| **循环网络** | RNN 与梯度问题、LSTM / GRU、seq2seq、注意力的起点（接注意力系列） | 待写 |
| **表示学习** | word2vec / GloVe、分词：BPE、WordPiece、SentencePiece、自编码器与 VAE、对比学习 | 待写 |
| **生成式基础** | GAN 原理与训练难点、DCGAN / CycleGAN / StyleGAN | 待写 |

### 3 · 大模型

Transformer 主线：结构 → 预训练 → 对齐 → 推理。共 9 个知识块、48 个知识点，已完成 17。

| 知识块 | 包含的知识点 | 状态 |
|---|---|---|
| [**注意力系列**](3-llm/01-attention/) | 动机、基础 A：q/k/v 与加权求和、基础 B：多头、基础 C：掩码与位置、手算一遍、矩阵与形状、softmax 与 √d 梯度推导、多头的内部、位置编码：正弦 / RoPE / ALiBi、从零实现、真实模型里的注意力、高效注意力：KV cache / GQA / FlashAttention、综合自测与小项目 | 已完成 |
| [**Transformer 完整模型**](3-llm/tf-01-block/) | 残差与 Pre/Post-LN、FFN 与 SwiGLU、Encoder / Decoder / Decoder-only、从零训练迷你 GPT | 已完成 |
| **现代结构演进** | MoE 稀疏专家、MLA 与 GQA、长上下文与位置外推、状态空间模型与 Mamba、线性注意力 | 待写 |
| **预训练模型家族** | BERT / RoBERTa、GPT-1/2/3、T5 与统一文本到文本、Scaling Laws 与 Chinchilla | 待写 |
| **数据与训练配方** | 数据采集、清洗、去重、分词词表设计、批量、学习率与训练稳定性、数据配比与课程学习、合成数据 | 待写 |
| **微调** | SFT 指令微调、Adapter / Prefix、LoRA / QLoRA、知识蒸馏 | 待写 |
| **对齐** | 奖励模型、RLHF 与 PPO、DPO 及变体、Constitutional AI 与 RLAIF、奖励黑客 | 待写 |
| **推理模型** | 思维链 CoT、自一致与搜索、o1 与测试时计算、DeepSeek-R1 与 GRPO | 待写 |
| **评测与安全** | 基准与污染、LLM-as-judge、幻觉、越狱与安全对齐 | 待写 |

### 4 · 模态与应用

把模型用到更多地方。共 9 个知识块、34 个知识点，已完成 0。

| 知识块 | 包含的知识点 | 状态 |
|---|---|---|
| **视觉 Transformer** | ViT、DeiT、Swin、DiT | 待写 |
| **扩散模型** | DDPM、潜空间扩散、条件生成与引导、Flow Matching、少步蒸馏 | 待写 |
| **世界模型与视频** | World Models、视频扩散、Sora、Genie / GameNGen | 待写 |
| **多模态** | CLIP、BLIP / Flamingo、LLaVA | 待写 |
| **语音与音频** | wav2vec 2.0 / HuBERT、Whisper、AudioLM / MusicGen | 待写 |
| **检索增强 RAG** | 向量检索、分块与重排、RAG 评测 | 待写 |
| **Agent 与工具** | ReAct、Toolformer 与函数调用、记忆与规划、多智能体 | 待写 |
| **图神经网络** | GCN、GraphSAGE、GAT、GIN 与 Graphormer | 待写 |
| **推荐系统** | Wide&Deep、YouTube DNN、DeepFM、DIN 与 PinSage | 待写 |

### 5 · 工程与系统

跑得起来、用得起。共 4 个知识块、17 个知识点，已完成 0。

| 知识块 | 包含的知识点 | 状态 |
|---|---|---|
| **分布式训练** | 数据并行、ZeRO 与 FSDP、张量并行、流水线并行、检查点与容错 | 待写 |
| **推理优化** | KV cache（已有）、量化：INT8 / AWQ / GPTQ、投机解码、连续批处理与 PagedAttention、剪枝与蒸馏 | 部分覆盖（KV cache、GQA、FlashAttention 已在注意力系列） |
| **部署与服务** | vLLM 与推理服务、延迟与吞吐、成本估算、观测与护栏 | 待写 |
| **实战项目** | 从零训练小 GPT、企业 RAG 系统、微调到部署流水线 | 待写 |
