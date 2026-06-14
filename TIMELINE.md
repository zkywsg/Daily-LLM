# TIMELINE

> 自动生成自各家族节点的 frontmatter。**请勿手工编辑。**
> 重新生成：`python3 scripts/generate_timeline.py`

| 年份 | 名字 | 家族 | 关键思想 | 路径 |
|------|------|------|---------|------|
| 1986 | **RNN** | `02-rnn-lstm` | 把上一时刻的隐状态接回当前时刻输入,用一组共享权重在时间上递推,任意长度序列被压进一个固定维度向量 | [02-rnn-lstm/01-rnn.md](02-rnn-lstm/01-rnn.md) |
| 1997 | **LSTM** | `02-rnn-lstm` | 用三道门 + 一条细胞状态高速公路绕开梯度连乘,让循环网络第一次能稳定学到 100+ 步的长依赖 | [02-rnn-lstm/02-lstm.md](02-rnn-lstm/02-lstm.md) |
| 1998 | **LeNet-5** | `01-cnn` | 把卷积+池化+全连接这套范式第一次系统化定义出来，在手写数字识别上跑通 | [01-cnn/01-lenet.md](01-cnn/01-lenet.md) |
| 2012 | **AlexNet** | `01-cnn` | 首次在 ImageNet 大规模数据集上端到端训练深层 CNN（5 conv + 3 fc），Top-5 错误率达到 15.3% | [01-cnn/02-alexnet.md](01-cnn/02-alexnet.md) |
| 2014 | **VGG** | `01-cnn` | 把网络深度做到 16/19 层、并把所有卷积统一成 3×3，证明深度本身就是性能来源 | [01-cnn/03-vgg.md](01-cnn/03-vgg.md) |
| 2014 | **GoogLeNet (Inception v1)** | `01-cnn` | 用 1×1 卷积降维 + 多尺度并行的 Inception 模块，把参数量压到 VGG 的 1/12 同时拿下 ImageNet 冠军 | [01-cnn/04-inception.md](01-cnn/04-inception.md) |
| 2014 | **GRU** | `02-rnn-lstm` | 把 LSTM 三道门简成两门、去掉细胞状态,参数减少 25% 而性能基本持平,成为 LSTM 的常用轻量替代 | [02-rnn-lstm/03-gru.md](02-rnn-lstm/03-gru.md) |
| 2014 | **Seq2Seq** | `02-rnn-lstm` | 用一个 encoder RNN 把任意长输入压成上下文向量,再用一个 decoder RNN 从这个向量生成任意长输出,统一所有序列到序列任务 | [02-rnn-lstm/04-seq2seq.md](02-rnn-lstm/04-seq2seq.md) |
| 2015 | **ResNet** | `01-cnn` | 用 shortcut 让网络只学残差修正而不是从零重建映射，把 152 层稳定训练变成可能 | [01-cnn/05-resnet.md](01-cnn/05-resnet.md) |
| 2015 | **Bahdanau Attention** | `02-rnn-lstm` | 在 Seq2Seq 上加 attention 让 decoder 每一步对 encoder 全部时刻学一个加权分布,绕开固定长度上下文向量的信息瓶颈 | [02-rnn-lstm/05-attention.md](02-rnn-lstm/05-attention.md) |
| 2017 | **DenseNet** | `01-cnn` | 每层都直接接收前面所有层的输出（concat 而非加法），把特征复用推到极致 | [01-cnn/06-densenet.md](01-cnn/06-densenet.md) |
| 2017 | **Transformer** | `05-transformer` | 用 self-attention 替代循环,让序列建模获得完全并行 + 全局上下文,encoder-decoder 骨架保留但内部全是 attention 和 FFN | [05-transformer/01-transformer.md](05-transformer/01-transformer.md) |
| 2018 | **BERT** | `06-bert-family` | 用 encoder-only Transformer + masked LM 学双向上下文表征,GLUE 11 任务全面 SOTA,把 NLP 拖进预训练时代 | [06-bert-family/01-bert.md](06-bert-family/01-bert.md) |
| 2018 | **GPT-1** | `07-gpt-scaling` | 用 decoder-only Transformer + 无监督自回归预训练 + 任务微调,第一次系统跑通预训练范式;同年 BERT 用 encoder-only 验证了双向版本 | [07-gpt-scaling/01-gpt1.md](07-gpt-scaling/01-gpt1.md) |
| 2019 | **EfficientNet** | `01-cnn` | 用复合缩放系数把 depth/width/resolution 三轴联合缩放公式化，得到帕累托最优的 B0–B7 模型族 | [01-cnn/07-efficientnet.md](01-cnn/07-efficientnet.md) |
| 2019 | **Transformer-XL** | `05-transformer` | 用段级循环把上一段隐状态作为这段的记忆 + 相对位置编码替代绝对 PE,让 Transformer 第一次跨越固定窗口处理长上下文 | [05-transformer/02-transformer-xl.md](05-transformer/02-transformer-xl.md) |
| 2019 | **RoBERTa** | `06-bert-family` | 去掉 NSP + 动态 masking + 大 batch + 10× 数据 + 更长训练,证明 BERT 严重训练不足,GLUE 再涨 5+ 分而架构完全不动 | [06-bert-family/02-roberta.md](06-bert-family/02-roberta.md) |
| 2019 | **ALBERT** | `06-bert-family` | 用跨层参数共享 + embedding 因式分解把 BERT-large 参数从 334M 压到 18M 而效果接近,同时把 NSP 改成更难的 SOP(句子顺序预测) | [06-bert-family/03-albert.md](06-bert-family/03-albert.md) |
| 2019 | **DistilBERT** | `06-bert-family` | 用知识蒸馏把 12 层 BERT teacher 压成 6 层 student,40% 参数 60% 速度保留 97% 性能,工业 BERT 部署的事实默认 | [06-bert-family/04-distilbert.md](06-bert-family/04-distilbert.md) |
| 2019 | **GPT-2** | `07-gpt-scaling` | 把 GPT-1 的 117M 参数推到 1.5B + WebText 40B token,zero-shot 任务能力首次涌现,LM 第一次显示出'不微调也能做下游任务'的通用性 | [07-gpt-scaling/02-gpt2.md](07-gpt-scaling/02-gpt2.md) |
| 2020 | **Sparse Attention** | `05-transformer` | 用滑窗局部 attention + 少量全局 token 把 attention 复杂度从 O(N²) 降到 O(N),让 Transformer 第一次能在 4K–16K 长上下文上跑训练和推理 | [05-transformer/03-sparse-attention.md](05-transformer/03-sparse-attention.md) |
| 2020 | **GPT-3** | `07-gpt-scaling` | 把 GPT-2 推到 175B 参数,in-context learning 涌现 — 仅靠 prompt 里 few-shot 例子就能学新任务,完全消除微调对监督数据的依赖,LLM 时代正式开启 | [07-gpt-scaling/03-gpt3.md](07-gpt-scaling/03-gpt3.md) |
| 2020 | **Scaling Laws** | `07-gpt-scaling` | 把 LM loss 随参数 N / 数据 D / 算力 C 的关系刻画成幂律;Kaplan 给出粗略最优,Chinchilla 修正最优配比是 N:D ≈ 1:20,催生 LLaMA 等高数据小模型 | [07-gpt-scaling/04-scaling-laws.md](07-gpt-scaling/04-scaling-laws.md) |
| 2020 | **ViT** | `08-vit` | 把图像切成 16×16 的 patch 当 token,用纯 Transformer encoder 处理,在 JFT-300M 上预训练后击败 CNN,证明视觉归纳偏置不是必需的 | [08-vit/01-vit.md](08-vit/01-vit.md) |
| 2020 | **DDPM** | `10-diffusion` | 把 2015 年的 diffusion 思想工程化:U-Net 预测噪声 + 简单 MSE 损失 + 1000 步去噪采样,稳定训练且质量超 GAN | [10-diffusion/01-ddpm.md](10-diffusion/01-ddpm.md) |
| 2020 | **Learning to Summarize from Human Feedback** | `12-rlhf-alignment` | 用人工偏好比较训练 reward model + PPO 微调 LLM,摘要质量超过监督学习 baseline 和参考摘要,确立 RLHF 在 NLP 上的完整方案 | [12-rlhf-alignment/01-learning-to-summarize.md](12-rlhf-alignment/01-learning-to-summarize.md) |
| 2021 | **RoPE** | `05-transformer` | 把位置信息编码进 Q/K 的旋转里而不是加在 token embedding 上,attention 内积天然只依赖相对位置,长上下文外推显著更好 | [05-transformer/04-rope.md](05-transformer/04-rope.md) |
| 2021 | **DeiT** | `08-vit` | 用 distillation token + 强增强 + AdamW + 蒸馏让 ViT 在 ImageNet-1K 上从零训练击败 ResNet,不再依赖 JFT-300M,把 ViT 带给学界 | [08-vit/02-deit.md](08-vit/02-deit.md) |
| 2021 | **Swin Transformer** | `08-vit` | 用 shifted window attention 把复杂度从 O(N²) 降到 O(N) + 层级化下采样产出多尺度特征图,让 ViT 第一次能直接做 detection / segmentation | [08-vit/03-swin.md](08-vit/03-swin.md) |
| 2021 | **CLIP** | `09-multimodal-clip` | 用 4 亿对网络图文数据做对比学习,让图像和文本编码到同一向量空间,zero-shot 分类直接匹配监督 SOTA;成为后续所有多模态系统的对齐基座 | [09-multimodal-clip/01-clip.md](09-multimodal-clip/01-clip.md) |
| 2022 | **ConvNeXt** | `01-cnn` | 把 ViT 的所有现代化设计选择（大 kernel·LayerNorm·GELU·强增强）逐项搬回 ResNet，CNN 反超 ViT | [01-cnn/08-convnext.md](01-cnn/08-convnext.md) |
| 2022 | **FlashAttention** | `05-transformer` | 把 attention 从 HBM 搬到 SRAM 算,分块 + 重计算把 O(N²) 显存压成 O(N) 而结果完全等价,attention 训练/推理快 2-4× 且支持更长序列 | [05-transformer/05-flash-attention.md](05-transformer/05-flash-attention.md) |
| 2022 | **DiT** | `08-vit` | 把 diffusion 模型的 U-Net backbone 替换成 ViT-style Transformer,展示更好的 scaling 性质,成为 Stable Diffusion 3 / Sora 的基座 | [08-vit/04-dit.md](08-vit/04-dit.md) |
| 2022 | **BLIP / BLIP-2** | `09-multimodal-clip` | 在 CLIP 对比之上加入生成和匹配两个任务联合训练;BLIP-2 进一步用 Q-Former 桥接冻结视觉编码器和冻结 LLM,把训练成本降一个数量级 | [09-multimodal-clip/02-blip.md](09-multimodal-clip/02-blip.md) |
| 2022 | **LDM / Stable Diffusion** | `10-diffusion` | 把 diffusion 从 pixel 空间移到 VAE latent 空间,推理显存降 64×,2022 年 8 月以开源方式释出 Stable Diffusion 把文生图带到消费 GPU | [10-diffusion/02-ldm.md](10-diffusion/02-ldm.md) |
| 2022 | **Imagen / Classifier-Free Guidance** | `10-diffusion` | 用大文本编码器(T5-XXL)+ classifier-free guidance,把文本理解和可控性推到 SOTA;CFG 成为所有现代 diffusion 模型的标配 | [10-diffusion/03-imagen.md](10-diffusion/03-imagen.md) |
| 2022 | **InstructGPT** | `12-rlhf-alignment` | 把 RLHF 三阶段从摘要单一任务推广到通用指令跟随,1.3B 对齐版超过 175B 未对齐版,直接催生 ChatGPT | [12-rlhf-alignment/02-instructgpt.md](12-rlhf-alignment/02-instructgpt.md) |
| 2022 | **Constitutional AI** | `12-rlhf-alignment` | 用一套书面原则(constitution)让 AI 自评自身输出,生成 AI feedback 替代人类偏好标注 — 把对齐从'人力密集'压成'算力密集',是 Claude 系列的核心方法 | [12-rlhf-alignment/03-constitutional-ai.md](12-rlhf-alignment/03-constitutional-ai.md) |
| 2023 | **GPT-4 / LLaMA** | `07-gpt-scaling` | GPT-4 把 LLM 推到万亿级 + 多模态闭源;LLaMA 给社区第一个工业级开源基础模型;现代 LLM 配方(Pre-RMSNorm + RoPE + GQA + SwiGLU)在两者上同时定型 | [07-gpt-scaling/05-gpt4-llama.md](07-gpt-scaling/05-gpt4-llama.md) |
| 2023 | **Flow Matching / Rectified Flow** | `10-diffusion` | 把 diffusion 的 ε-prediction 推广到任意流形的'速度场学习',训练更稳 + 采样路径更直 + 数学更简洁,SD3 / Flux 默认 | [10-diffusion/04-flow-matching.md](10-diffusion/04-flow-matching.md) |
| 2023 | **DPO** | `12-rlhf-alignment` | 通过数学推导把 RLHF 的 RL 目标转化成监督学习损失,跳过 reward model 和 PPO,工程上和 SFT 一样简单且效果接近,2024 开源 LLM 默认对齐方法 | [12-rlhf-alignment/04-dpo.md](12-rlhf-alignment/04-dpo.md) |
