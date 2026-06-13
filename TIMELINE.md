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
| 2019 | **EfficientNet** | `01-cnn` | 用复合缩放系数把 depth/width/resolution 三轴联合缩放公式化，得到帕累托最优的 B0–B7 模型族 | [01-cnn/07-efficientnet.md](01-cnn/07-efficientnet.md) |
| 2019 | **Transformer-XL** | `05-transformer` | 用段级循环把上一段隐状态作为这段的记忆 + 相对位置编码替代绝对 PE,让 Transformer 第一次跨越固定窗口处理长上下文 | [05-transformer/02-transformer-xl.md](05-transformer/02-transformer-xl.md) |
| 2020 | **Sparse Attention** | `05-transformer` | 用滑窗局部 attention + 少量全局 token 把 attention 复杂度从 O(N²) 降到 O(N),让 Transformer 第一次能在 4K–16K 长上下文上跑训练和推理 | [05-transformer/03-sparse-attention.md](05-transformer/03-sparse-attention.md) |
| 2021 | **RoPE** | `05-transformer` | 把位置信息编码进 Q/K 的旋转里而不是加在 token embedding 上,attention 内积天然只依赖相对位置,长上下文外推显著更好 | [05-transformer/04-rope.md](05-transformer/04-rope.md) |
| 2022 | **ConvNeXt** | `01-cnn` | 把 ViT 的所有现代化设计选择（大 kernel·LayerNorm·GELU·强增强）逐项搬回 ResNet，CNN 反超 ViT | [01-cnn/08-convnext.md](01-cnn/08-convnext.md) |
