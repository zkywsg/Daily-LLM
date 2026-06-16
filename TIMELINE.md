# TIMELINE

> 自动生成自各家族节点的 frontmatter。**请勿手工编辑。**
> 重新生成：`python3 scripts/generate_timeline.py`

| 年份 | 名字 | 家族 | 关键思想 | 路径 |
|------|------|------|---------|------|
| 1986 | **RNN** | `02-rnn-lstm` | 把上一时刻的隐状态接回当前时刻输入,用一组共享权重在时间上递推,任意长度序列被压进一个固定维度向量 | [02-rnn-lstm/01-rnn.md](02-rnn-lstm/01-rnn.md) |
| 1997 | **LSTM** | `02-rnn-lstm` | 用三道门 + 一条细胞状态高速公路绕开梯度连乘,让循环网络第一次能稳定学到 100+ 步的长依赖 | [02-rnn-lstm/02-lstm.md](02-rnn-lstm/02-lstm.md) |
| 1998 | **LeNet-5** | `01-cnn` | 把卷积+池化+全连接这套范式第一次系统化定义出来，在手写数字识别上跑通 | [01-cnn/01-lenet.md](01-cnn/01-lenet.md) |
| 2012 | **AlexNet** | `01-cnn` | 首次在 ImageNet 大规模数据集上端到端训练深层 CNN（5 conv + 3 fc），Top-5 错误率达到 15.3% | [01-cnn/02-alexnet.md](01-cnn/02-alexnet.md) |
| 2013 | **Word2Vec** | `03-word-embedding` | 用极简的 shallow network(skip-gram / CBOW)从大语料自监督学 300 维词向量,negative sampling 替代 softmax 让训练扩展到 100 亿词;king-man+woman=queen 线性算术成立,NLP 进入向量时代 | [03-word-embedding/01-word2vec.md](03-word-embedding/01-word2vec.md) |
| 2014 | **VGG** | `01-cnn` | 把网络深度做到 16/19 层、并把所有卷积统一成 3×3，证明深度本身就是性能来源 | [01-cnn/03-vgg.md](01-cnn/03-vgg.md) |
| 2014 | **GoogLeNet (Inception v1)** | `01-cnn` | 用 1×1 卷积降维 + 多尺度并行的 Inception 模块，把参数量压到 VGG 的 1/12 同时拿下 ImageNet 冠军 | [01-cnn/04-inception.md](01-cnn/04-inception.md) |
| 2014 | **GRU** | `02-rnn-lstm` | 把 LSTM 三道门简成两门、去掉细胞状态,参数减少 25% 而性能基本持平,成为 LSTM 的常用轻量替代 | [02-rnn-lstm/03-gru.md](02-rnn-lstm/03-gru.md) |
| 2014 | **Seq2Seq** | `02-rnn-lstm` | 用一个 encoder RNN 把任意长输入压成上下文向量,再用一个 decoder RNN 从这个向量生成任意长输出,统一所有序列到序列任务 | [02-rnn-lstm/04-seq2seq.md](02-rnn-lstm/04-seq2seq.md) |
| 2014 | **GloVe** | `03-word-embedding` | 直接对全局 word-word 共现矩阵做加权 log-bilinear 分解,而不是用局部窗口预测;count-based 路线,理论比 Word2Vec 更清晰,Stanford 团队预训练好的 GloVe 向量成开源标配 | [03-word-embedding/02-glove.md](03-word-embedding/02-glove.md) |
| 2014 | **GAN** | `04-gan` | 把生成问题转化为对抗博弈——G 造假,D 鉴别,minimax 训练让 G 学到真实数据分布;不显式建模 likelihood 也能生成高质量样本,开创深度生成模型新范式 | [04-gan/01-gan.md](04-gan/01-gan.md) |
| 2015 | **ResNet** | `01-cnn` | 用 shortcut 让网络只学残差修正而不是从零重建映射，把 152 层稳定训练变成可能 | [01-cnn/05-resnet.md](01-cnn/05-resnet.md) |
| 2015 | **Bahdanau Attention** | `02-rnn-lstm` | 在 Seq2Seq 上加 attention 让 decoder 每一步对 encoder 全部时刻学一个加权分布,绕开固定长度上下文向量的信息瓶颈 | [02-rnn-lstm/05-attention.md](02-rnn-lstm/05-attention.md) |
| 2015 | **DCGAN** | `04-gan` | 把 CNN 完整移植到 GAN——用 strided conv 替代 pooling、加 BatchNorm、Generator 用 transpose conv 上采样、去全连接层;首次给出可复现的 GAN 训练工程方案,生成 64×64 卧室 / 人脸图像 | [04-gan/02-dcgan.md](04-gan/02-dcgan.md) |
| 2016 | **FastText** | `03-word-embedding` | 把词拆成 character n-gram(apple = <ap, app, ppl, ple, le>),词向量 = subword 向量之和;处理 OOV / 形态丰富语言 / 罕见词;同时附带极快的文本分类工具 | [03-word-embedding/03-fasttext.md](03-word-embedding/03-fasttext.md) |
| 2017 | **DenseNet** | `01-cnn` | 每层都直接接收前面所有层的输出（concat 而非加法），把特征复用推到极致 | [01-cnn/06-densenet.md](01-cnn/06-densenet.md) |
| 2017 | **CycleGAN** | `04-gan` | 用 cycle consistency loss 实现无配对图像翻译——两个 G 互相 mapping(X→Y 和 Y→X),要求 F(G(x)) ≈ x;不需要成对训练数据就能做马↔斑马、夏↔冬、照片↔画风的转换 | [04-gan/03-cyclegan.md](04-gan/03-cyclegan.md) |
| 2017 | **Transformer** | `05-transformer` | 用 self-attention 替代循环,让序列建模获得完全并行 + 全局上下文,encoder-decoder 骨架保留但内部全是 attention 和 FFN | [05-transformer/01-transformer.md](05-transformer/01-transformer.md) |
| 2017 | **Outrageously Large Neural Networks (Sparsely-Gated MoE)** | `13-moe-efficient` | 在 LSTM 之间插入 sparsely-gated MoE 层:每 token 用 gate 选 top-K 个 expert(1370 亿参数中只激活几亿),配 auxiliary loss 防止 expert 塌缩;首次证明稀疏激活能突破 dense 模型的参数 / 算力锁死 | [13-moe-efficient/01-sparsely-gated-moe.md](13-moe-efficient/01-sparsely-gated-moe.md) |
| 2018 | **ELMo** | `03-word-embedding` | 用双向 LSTM 预训练语言模型,每个词的向量是 LSTM 各层 hidden state 的加权和;同一个 \"bank\" 在 \"river bank\" 和 \"money bank\" 里向量不同;contextualized embedding 起源,直接催生 BERT | [03-word-embedding/04-elmo.md](03-word-embedding/04-elmo.md) |
| 2018 | **StyleGAN** | `04-gan` | 用 mapping network 把 z 投射到 W 空间,再通过 AdaIN 在每层注入 style 控制不同语义粒度(粗:姿态/形状,中:发型/眼神,细:肤色/纹理);1024×1024 超高分辨率人脸,生成质量逼近真实照片 | [04-gan/04-stylegan.md](04-gan/04-stylegan.md) |
| 2018 | **BERT** | `06-bert-family` | 用 encoder-only Transformer + masked LM 学双向上下文表征,GLUE 11 任务全面 SOTA,把 NLP 拖进预训练时代 | [06-bert-family/01-bert.md](06-bert-family/01-bert.md) |
| 2018 | **GPT-1** | `07-gpt-scaling` | 用 decoder-only Transformer + 无监督自回归预训练 + 任务微调,第一次系统跑通预训练范式;同年 BERT 用 encoder-only 验证了双向版本 | [07-gpt-scaling/01-gpt1.md](07-gpt-scaling/01-gpt1.md) |
| 2019 | **EfficientNet** | `01-cnn` | 用复合缩放系数把 depth/width/resolution 三轴联合缩放公式化，得到帕累托最优的 B0–B7 模型族 | [01-cnn/07-efficientnet.md](01-cnn/07-efficientnet.md) |
| 2019 | **Transformer-XL** | `05-transformer` | 用段级循环把上一段隐状态作为这段的记忆 + 相对位置编码替代绝对 PE,让 Transformer 第一次跨越固定窗口处理长上下文 | [05-transformer/02-transformer-xl.md](05-transformer/02-transformer-xl.md) |
| 2019 | **RoBERTa** | `06-bert-family` | 去掉 NSP + 动态 masking + 大 batch + 10× 数据 + 更长训练,证明 BERT 严重训练不足,GLUE 再涨 5+ 分而架构完全不动 | [06-bert-family/02-roberta.md](06-bert-family/02-roberta.md) |
| 2019 | **ALBERT** | `06-bert-family` | 用跨层参数共享 + embedding 因式分解把 BERT-large 参数从 334M 压到 18M 而效果接近,同时把 NSP 改成更难的 SOP(句子顺序预测) | [06-bert-family/03-albert.md](06-bert-family/03-albert.md) |
| 2019 | **DistilBERT** | `06-bert-family` | 用知识蒸馏把 12 层 BERT teacher 压成 6 层 student,40% 参数 60% 速度保留 97% 性能,工业 BERT 部署的事实默认 | [06-bert-family/04-distilbert.md](06-bert-family/04-distilbert.md) |
| 2019 | **GPT-2** | `07-gpt-scaling` | 把 GPT-1 的 117M 参数推到 1.5B + WebText 40B token,zero-shot 任务能力首次涌现,LM 第一次显示出'不微调也能做下游任务'的通用性 | [07-gpt-scaling/02-gpt2.md](07-gpt-scaling/02-gpt2.md) |
| 2019 | **Adapter Tuning** | `11-peft-lora` | 在每层 Transformer 插入 small bottleneck adapter 模块(down → ReLU → up + residual),base 模型完全冻结,只训 3% 参数达到全参微调 96% 性能;PEFT 起源,后续 LoRA / Prefix Tuning 都受其启发 | [11-peft-lora/01-adapter.md](11-peft-lora/01-adapter.md) |
| 2020 | **Sparse Attention** | `05-transformer` | 用滑窗局部 attention + 少量全局 token 把 attention 复杂度从 O(N²) 降到 O(N),让 Transformer 第一次能在 4K–16K 长上下文上跑训练和推理 | [05-transformer/03-sparse-attention.md](05-transformer/03-sparse-attention.md) |
| 2020 | **GPT-3** | `07-gpt-scaling` | 把 GPT-2 推到 175B 参数,in-context learning 涌现 — 仅靠 prompt 里 few-shot 例子就能学新任务,完全消除微调对监督数据的依赖,LLM 时代正式开启 | [07-gpt-scaling/03-gpt3.md](07-gpt-scaling/03-gpt3.md) |
| 2020 | **Scaling Laws** | `07-gpt-scaling` | 把 LM loss 随参数 N / 数据 D / 算力 C 的关系刻画成幂律;Kaplan 给出粗略最优,Chinchilla 修正最优配比是 N:D ≈ 1:20,催生 LLaMA 等高数据小模型 | [07-gpt-scaling/04-scaling-laws.md](07-gpt-scaling/04-scaling-laws.md) |
| 2020 | **ViT** | `08-vit` | 把图像切成 16×16 的 patch 当 token,用纯 Transformer encoder 处理,在 JFT-300M 上预训练后击败 CNN,证明视觉归纳偏置不是必需的 | [08-vit/01-vit.md](08-vit/01-vit.md) |
| 2020 | **DDPM** | `10-diffusion` | 把 2015 年的 diffusion 思想工程化:U-Net 预测噪声 + 简单 MSE 损失 + 1000 步去噪采样,稳定训练且质量超 GAN | [10-diffusion/01-ddpm.md](10-diffusion/01-ddpm.md) |
| 2020 | **Learning to Summarize from Human Feedback** | `12-rlhf-alignment` | 用人工偏好比较训练 reward model + PPO 微调 LLM,摘要质量超过监督学习 baseline 和参考摘要,确立 RLHF 在 NLP 上的完整方案 | [12-rlhf-alignment/01-learning-to-summarize.md](12-rlhf-alignment/01-learning-to-summarize.md) |
| 2020 | **RAG** | `14-rag-agent` | 把 dense retriever(DPR)和 seq2seq 生成器联合训练,把外部知识库接进 LM 输入侧;开放域 QA 不再依赖参数化知识,可以查 | [14-rag-agent/01-rag.md](14-rag-agent/01-rag.md) |
| 2021 | **RoPE** | `05-transformer` | 把位置信息编码进 Q/K 的旋转里而不是加在 token embedding 上,attention 内积天然只依赖相对位置,长上下文外推显著更好 | [05-transformer/04-rope.md](05-transformer/04-rope.md) |
| 2021 | **DeiT** | `08-vit` | 用 distillation token + 强增强 + AdamW + 蒸馏让 ViT 在 ImageNet-1K 上从零训练击败 ResNet,不再依赖 JFT-300M,把 ViT 带给学界 | [08-vit/02-deit.md](08-vit/02-deit.md) |
| 2021 | **Swin Transformer** | `08-vit` | 用 shifted window attention 把复杂度从 O(N²) 降到 O(N) + 层级化下采样产出多尺度特征图,让 ViT 第一次能直接做 detection / segmentation | [08-vit/03-swin.md](08-vit/03-swin.md) |
| 2021 | **CLIP** | `09-multimodal-clip` | 用 4 亿对网络图文数据做对比学习,让图像和文本编码到同一向量空间,zero-shot 分类直接匹配监督 SOTA;成为后续所有多模态系统的对齐基座 | [09-multimodal-clip/01-clip.md](09-multimodal-clip/01-clip.md) |
| 2021 | **Prefix Tuning** | `11-peft-lora` | 在每层 attention 的 K/V 前面加一段可学习的"soft prefix" embedding,base 模型完全冻结,只训这段 prefix(~0.1% 参数);极致参数效率,1000+ task 用同一 base 共享 | [11-peft-lora/02-prefix-tuning.md](11-peft-lora/02-prefix-tuning.md) |
| 2021 | **LoRA** | `11-peft-lora` | 把权重更新 ΔW 分解为低秩矩阵 B·A(r 远小于 d),只训 BA 的 ~0.1% 参数;推理时 W = W₀ + BA 可合并回原权重,零额外延迟;PEFT 时代的工业标准 | [11-peft-lora/03-lora.md](11-peft-lora/03-lora.md) |
| 2021 | **Switch Transformer** | `13-moe-efficient` | 把 MoE 移植到 Transformer + 简化为 top-1 gating(每 token 只走一个 expert,代替 Shazeer top-K),配 load balancing loss 和 selective precision;首次做到 1.6T 参数模型,T5-XXL 4× 加速同质量 | [13-moe-efficient/02-switch-transformer.md](13-moe-efficient/02-switch-transformer.md) |
| 2022 | **ConvNeXt** | `01-cnn` | 把 ViT 的所有现代化设计选择（大 kernel·LayerNorm·GELU·强增强）逐项搬回 ResNet，CNN 反超 ViT | [01-cnn/08-convnext.md](01-cnn/08-convnext.md) |
| 2022 | **FlashAttention** | `05-transformer` | 把 attention 从 HBM 搬到 SRAM 算,分块 + 重计算把 O(N²) 显存压成 O(N) 而结果完全等价,attention 训练/推理快 2-4× 且支持更长序列 | [05-transformer/05-flash-attention.md](05-transformer/05-flash-attention.md) |
| 2022 | **DiT** | `08-vit` | 把 diffusion 模型的 U-Net backbone 替换成 ViT-style Transformer,展示更好的 scaling 性质,成为 Stable Diffusion 3 / Sora 的基座 | [08-vit/04-dit.md](08-vit/04-dit.md) |
| 2022 | **BLIP / BLIP-2** | `09-multimodal-clip` | 在 CLIP 对比之上加入生成和匹配两个任务联合训练;BLIP-2 进一步用 Q-Former 桥接冻结视觉编码器和冻结 LLM,把训练成本降一个数量级 | [09-multimodal-clip/02-blip.md](09-multimodal-clip/02-blip.md) |
| 2022 | **Flamingo** | `09-multimodal-clip` | 冻结大 LLM(Chinchilla 70B)+ Perceiver Resampler 视觉适配 + 间隔 cross-attention 注入,8 例 in-context 学新视觉任务的少样本 VLM 范式 | [09-multimodal-clip/03-flamingo.md](09-multimodal-clip/03-flamingo.md) |
| 2022 | **LDM / Stable Diffusion** | `10-diffusion` | 把 diffusion 从 pixel 空间移到 VAE latent 空间,推理显存降 64×,2022 年 8 月以开源方式释出 Stable Diffusion 把文生图带到消费 GPU | [10-diffusion/02-ldm.md](10-diffusion/02-ldm.md) |
| 2022 | **Imagen / Classifier-Free Guidance** | `10-diffusion` | 用大文本编码器(T5-XXL)+ classifier-free guidance,把文本理解和可控性推到 SOTA;CFG 成为所有现代 diffusion 模型的标配 | [10-diffusion/03-imagen.md](10-diffusion/03-imagen.md) |
| 2022 | **InstructGPT** | `12-rlhf-alignment` | 把 RLHF 三阶段从摘要单一任务推广到通用指令跟随,1.3B 对齐版超过 175B 未对齐版,直接催生 ChatGPT | [12-rlhf-alignment/02-instructgpt.md](12-rlhf-alignment/02-instructgpt.md) |
| 2022 | **Constitutional AI** | `12-rlhf-alignment` | 用一套书面原则(constitution)让 AI 自评自身输出,生成 AI feedback 替代人类偏好标注 — 把对齐从'人力密集'压成'算力密集',是 Claude 系列的核心方法 | [12-rlhf-alignment/03-constitutional-ai.md](12-rlhf-alignment/03-constitutional-ai.md) |
| 2022 | **ReAct** | `14-rag-agent` | 把 LLM 的推理(Thought)和行动(Action)交错进行,thought 推理下一步要查什么,action 调外部工具,observation 反馈给 LLM 继续推理;Agent 范式的起源 | [14-rag-agent/02-react.md](14-rag-agent/02-react.md) |
| 2022 | **Chain-of-Thought** | `15-reasoning-o1-r1` | 在 prompt 里给 few-shot 例子展示'问题→推理步骤→答案'格式,LLM 模仿后大数学题准确率从 17% 涨到 60%+;开启 LLM 推理能力的新研究方向 | [15-reasoning-o1-r1/01-cot.md](15-reasoning-o1-r1/01-cot.md) |
| 2022 | **Self-Consistency** | `15-reasoning-o1-r1` | 对同 prompt 采样 N 条 CoT 推理路径,投票选最一致答案;GSM8K 60% → 75%;第一次系统化 test-time compute scaling | [15-reasoning-o1-r1/02-self-consistency.md](15-reasoning-o1-r1/02-self-consistency.md) |
| 2023 | **GPT-4 / LLaMA** | `07-gpt-scaling` | GPT-4 把 LLM 推到万亿级 + 多模态闭源;LLaMA 给社区第一个工业级开源基础模型;现代 LLM 配方(Pre-RMSNorm + RoPE + GQA + SwiGLU)在两者上同时定型 | [07-gpt-scaling/05-gpt4-llama.md](07-gpt-scaling/05-gpt4-llama.md) |
| 2023 | **LLaVA** | `09-multimodal-clip` | Visual instruction tuning:用 GPT-4 自动生成视觉指令数据,把 CLIP 视觉特征用单 linear projection 接到 LLaMA,把开源 VLM 范式定型在 GPT-4V 之前 | [09-multimodal-clip/04-llava.md](09-multimodal-clip/04-llava.md) |
| 2023 | **Flow Matching / Rectified Flow** | `10-diffusion` | 把 diffusion 的 ε-prediction 推广到任意流形的'速度场学习',训练更稳 + 采样路径更直 + 数学更简洁,SD3 / Flux 默认 | [10-diffusion/04-flow-matching.md](10-diffusion/04-flow-matching.md) |
| 2023 | **QLoRA** | `11-peft-lora` | Base 模型量化到 4-bit NF4(NormalFloat)+ LoRA 微调,配 double quantization + paged optimizer;让 65B 模型能在单卡 48GB(实际 24GB 也能)GPU 上微调,LLM 微调彻底个人化 | [11-peft-lora/04-qlora.md](11-peft-lora/04-qlora.md) |
| 2023 | **DPO** | `12-rlhf-alignment` | 通过数学推导把 RLHF 的 RL 目标转化成监督学习损失,跳过 reward model 和 PPO,工程上和 SFT 一样简单且效果接近,2024 开源 LLM 默认对齐方法 | [12-rlhf-alignment/04-dpo.md](12-rlhf-alignment/04-dpo.md) |
| 2023 | **Toolformer** | `14-rag-agent` | 让 LLM 在预训练语料上自监督学习何时何处插入工具调用——给候选位置加 tool call,如果调用后 perplexity 降低就保留;tool use 从 prompt 技巧内化为模型本身能力 | [14-rag-agent/03-toolformer.md](14-rag-agent/03-toolformer.md) |
| 2023 | **AutoGPT** | `14-rag-agent` | 把 ReAct 推到极限——LLM 拿到高级目标后自己分解为子任务、规划执行步骤、循环调工具直到完成,无人干预;启动自主 agent 范式 | [14-rag-agent/04-autogpt.md](14-rag-agent/04-autogpt.md) |
| 2024 | **Mixtral 8×7B** | `13-moe-efficient` | 第一个完全开源的生产级 MoE LLM,8 个 7B expert + top-2 gating,46.7B 总参 / 13B 激活;质量超 LLaMA-2-70B 但推理速度像 13B 模型,开源社区第一次拿到可用 MoE | [13-moe-efficient/03-mixtral.md](13-moe-efficient/03-mixtral.md) |
| 2024 | **DeepSeek-V3** | `13-moe-efficient` | 671B 总参 / 37B 激活的开源 MoE 旗舰,集成 fine-grained experts(256 细粒度 expert)+ shared experts + aux-loss-free load balancing + MTP(Multi-Token Prediction)等十余项创新;首次让开源 MoE 追上 GPT-4 级闭源模型,也是 DeepSeek-R1 的 base | [13-moe-efficient/04-deepseek-v3.md](13-moe-efficient/04-deepseek-v3.md) |
| 2024 | **OpenAI o1** | `15-reasoning-o1-r1` | 把长链推理作为训练目标,用 RL 让 LLM 自己学到反思/回溯/自验证;test-time compute 成为继训练算力之后的新 scaling 轴,在数学/科学/代码 benchmark 上击败 GPT-4 多倍 | [15-reasoning-o1-r1/03-o1.md](15-reasoning-o1-r1/03-o1.md) |
| 2025 | **DeepSeek-R1** | `15-reasoning-o1-r1` | 开源 o1 风格推理模型;先用纯 RL(GRPO)无 SFT cold start 训练 R1-Zero 验证推理行为可从 RL 中涌现,再用少量 cold-start SFT + 多阶段 RL 训练 R1 达到 o1 同级性能,推理 trace 全公开 | [15-reasoning-o1-r1/04-deepseek-r1.md](15-reasoning-o1-r1/04-deepseek-r1.md) |
