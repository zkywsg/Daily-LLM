---
name: "RNN"
year: 1986
family: "02-rnn-lstm"
order: 1
paper: "Learning Internal Representations by Error Propagation / Finding Structure in Time"
authors: ["David Rumelhart", "Geoffrey Hinton", "Ronald Williams", "Jeffrey Elman"]
key_idea: "把上一时刻的隐状态接回当前时刻输入,用一组共享权重在时间上递推,任意长度序列被压进一个固定维度向量"
---

# RNN (1986)

## 前作进展

1980 年代中期神经网络刚从 1969 年 Minsky & Papert《Perceptrons》引发的"AI 寒冬"中爬出来。这次复兴的核心事件是 1986 年 Rumelhart、Hinton、Williams 在 *Nature* 上发表《Learning Representations by Back-propagating Errors》,把反向传播算法系统化地引入多层神经网络——这是连续函数逼近问题的转折点,后人称之为"反传革命"。同期 Hopfield 在 1982 年提出 Hopfield 网络,Hinton 与 Sejnowski 在 1985 年提出 Boltzmann 机,这两条线都引入了"网络存在动态/状态"的思想,但它们的"状态"是为了求解能量极小化或采样,不是为了处理序列。

当时主流的神经网络是 **MLP(多层感知机)**:输入层、隐藏层、输出层全连接,信号从输入端单向流到输出端。MLP 在分类、回归这类"固定大小输入 → 固定大小输出"的任务上工作得很好,但语言、语音、时间序列这些**变长**的数据进不了 MLP。当时主流的解法有两类,各有硬伤:

- **时延神经网络(TDNN, Waibel 1989)**:把一个滑窗内的 N 个时间步拼成一个长向量喂给 MLP。优点是直接复用现有反传机器;缺点是窗口长度必须事先固定,跨窗口的依赖完全看不到,等同于"用 MLP 假装在做序列"
- **统计语言模型(n-gram)**:`P(w_t | w_{t-1}, ..., w_{t-n+1})`。靠概率表存所有 n-gram 频次,n=3 时词表 50k 就需要 1.25×10^14 个条目,组合爆炸限制了上下文长度只能到 3–5

两条路的共性问题是——**序列的"历史"被强行截断到一个固定长度窗口**。一句话里的"它"指代 20 个词之前的某个名词,或一段对话里的语义跨越多轮,这些跨距长的依赖根本进不了模型。

序列建模需要一个完全不同的范式:**模型自己维护一个内部状态,这个状态在读取每一个新输入时被更新,从而把"历史信息"压在状态里带着走**。

## 核心思想

### 直觉:把隐状态当成"沿时间传递的信息瓶颈"

理解 RNN 真正需要先抓一件事:**MLP 只能处理固定大小输入,语言 / 语音 / 时间序列这些变长数据完全进不去**。之前的解法 TDNN(滑窗 MLP)和 n-gram(概率表)都靠"固定窗口"硬截断,跨距长的依赖根本看不到。Jordan(1986)和 Elman(1990)反问:**为什么不让网络维护一个"内部状态",每读一个新输入就更新它,把所有历史信息压在状态里一直带着走?**

三件事必须同时成立才让 RNN 在 1986 年成立:

- **隐状态当固定维度的"信息瓶颈"** — 任意长度序列都压进一个 d 维向量(典型 100-500),模型必须自己学"什么该记什么该忘"
- **同一组权重在所有时间步共享** — `(W_x, W_h, W_y)` 不随 t 变化,等价于 CNN 卷积核在空间上扫,这里换成"时间上扫"。参数量与序列长度无关
- **BPTT(按时间反传)让训练可行** — 把循环按时间展开成深度 T 的前馈网络,跑标准反传

三件事合起来:RNN 第一次让"变长序列建模"在神经网络框架内统一表达。但 BPTT 反传时遇到致命问题 —— **梯度连乘 T 次,要么消失要么爆炸**,这是 1994 Bengio 论文证明的结构性缺陷,也是 1997 LSTM 用"细胞状态高速路"解决它的根本动机。

![RNN 按时间展开 + 隐状态递推](assets/01-rnn-unfold.svg)
*图 1:**上** RNN cell 的循环表示 — 隐状态 h_t 接收当前输入 x_t 和上一时刻 h_{t-1},经 tanh 得到新状态,输出 y_t。**下** 按时间展开 — 4 个时间步共享同一组 (W_x, W_h, W_y),h_t 沿时间链向前传递。展开后等价于深度 T 的前馈网络,可以跑标准反传(BPTT)。底部公式 `h_t = tanh(W_x x_t + W_h h_{t-1})`,callout 强调"参数共享让序列长度可任意但参数量不变"。*

### 机制一:隐状态递推 — h_t = f(W_x x_t + W_h h_{t-1})

RNN 的全部数学就两行:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b_h)
$$

$$
y_t = g(W_y h_t + b_y)
$$

`x_t` 是 t 时刻输入(如词 embedding),`h_t` 是 t 时刻隐状态,`y_t` 是 t 时刻输出。`g` 取 softmax(分类)或恒等(回归)。每个时间步:**新状态 = 旧状态 + 新输入 经一个非线性变换**。

**Jordan vs Elman 的区别仅在"反馈源"** — Jordan 网络反馈上一时刻的**输出** `y_{t-1}`,Elman 反馈上一时刻的**隐状态** `h_{t-1}`。Elman 方案胜出 —— 隐状态比输出维度更高、信息更丰富,后续所有循环网络都沿用 Elman 路线。

### 机制二:权重共享 — 同一组 (W_x, W_h, W_y) 复用所有 T 步

RNN 的所有时间步用同一组权重 `(W_x, W_h, W_y)`,这是它能处理变长序列的根本。**和 CNN 卷积核在空间上扫完全同构,只是共享方向从"空间"换成了"时间"**。

这一设计的两个直接后果:

- **参数量与序列长度无关** — 不管序列是 10 步还是 1000 步,模型参数都是固定的 `d² + d·input_dim + d·output_dim`
- **训练 / 推理时序列长度可以不同** — 训练时见过 50 步,推理时可以输入 200 步,只是隐状态质量未必跟得上

这条"权重共享让模型尺寸独立于序列长度"的设计后来被 Transformer 完全继承 —— 不管输入 token 是 100 还是 100K,Transformer 的参数都不变(虽然 attention 的 KV cache 与长度成正比,那是 inference 时的工程问题)。

### 机制三:BPTT — 按时间展开后跑标准反传

RNN 怎么训?Rumelhart 等人 1986 反传论文已经隐含给出答案:**把循环按时间展开**。考虑长度 T 的序列,把同一个 RNN cell 复制 T 份首尾相接,就变成深度 T 的前馈网络(每一层参数都是同一组 `(W_x, W_h)`)。在展开图上跑标准反传 = **BPTT(Backpropagation Through Time)**。

具体梯度推导:`L` 对 `h_1` 的梯度是

$$
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T-1} \frac{\partial h_{t+1}}{\partial h_t} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T-1} W_h^\top \, \text{diag}(\tanh'(h_t))
$$

这个乘积里 `tanh'(h_t)` 值域 (0, 1],而且越饱和越接近 0。**每一步乘一个谱半径接近 1 的矩阵 + 一个 (0,1] 的导数**,乘 T 次:

- `W_h` 谱半径 < 1 → 梯度按指数衰减 → **梯度消失**(loss 对早期 h_1 的梯度趋近 0,长依赖学不到)
- `W_h` 谱半径 > 1 → 梯度按指数放大 → **梯度爆炸**(数值上溢,训练 NaN)

工程上对梯度爆炸的实用补丁是 **gradient clipping** —— 梯度范数超阈值 τ(典型 5 或 10)就缩放回去。这招至今 Transformer 训练里都在用。但**梯度消失没有简单的工程补丁**,这是结构性问题,逼出了 LSTM 的细胞状态高速路设计。

![BPTT 梯度连乘 → 消失 / 爆炸](assets/01-rnn-bptt-gradient.svg)
*图 2:**上半** BPTT 梯度回传路径 — loss 对 h_1 的梯度经过 4 次 ∂h_{t+1}/∂h_t 连乘,每次乘 W_h · tanh'。**中** 三种情况:谱半径 <1 梯度按指数衰减(红色衰减曲线),谱半径 >1 梯度按指数爆炸(红色爆炸曲线),谱半径 ≈1 + 正交初始化(绿色平稳曲线)。**下** Bengio 1994 结论:T > 10-20 后简单 RNN 学不到长依赖,这是结构问题不是优化技巧能解决的;直接催生 LSTM 1997 的 cell highway 设计。*

### 三件套协同:隐状态递推 + 权重共享 + BPTT 缺一不可

RNN 在 1986 年成立为序列建模的通用范式,**不是单一改进**,而是三件套同时调到协同点 —— 任何一个抽掉 RNN 都不成立,这一点和 [ResNet](../01-cnn/05-resnet.md) 的 `shortcut + BN + He 初始化` 协同关系一致:

- **只有隐状态递推,没有权重共享** — 每个时间步用不同的 W_t,模型参数随序列长度线性增长,变长序列根本训不起;退化成"为每个长度训一个 MLP"
- **只有权重共享,没有隐状态递推** — 退化成 TDNN(滑窗 MLP),跨窗口的信息看不到,n-gram 那一类硬截断的所有问题都回来了
- **只有递推 + 共享,没有 BPTT** — 训练算法上无法把"循环结构"对应到反传,RNN 在理论上能跑但实际学不出来。BPTT 是让这一切可落地的算法基础

三件套合起来才让"用神经网络做序列建模"在 1986 年第一次成立。但它也留下两个明确遗憾 —— **梯度消失/爆炸**(由 [LSTM](02-lstm.md) 1997 用 cell highway 解决)和 **串行不可并行**(由 [Transformer](../05-transformer/01-transformer.md) 2017 用 self-attention 彻底放弃循环来解决)。

## 两个早期任务

简单 RNN 在 1990 年代被用在两类任务上,做出了一些当时看来惊艳的结果:

**Elman 1990 任务**——给一个由 2–3 词句子拼成的长字符串,让 RNN 逐字符预测下一个字符。Elman 发现:RNN 在没有任何监督信号告诉它"哪里是词边界"的情况下,**预测误差的曲线在词与词之间出现尖峰**——也就是说,RNN 自己学到了"词是一个有结构的单元"。这是神经网络第一次被证明可以从原始序列里**无监督地发现语言结构**,在当时(1990)是个相当强的论断。

**Mikolov 2010 RNN 语言模型**——这是简单 RNN 的工程化应用,把 RNN 用作 n-gram 的连续替代:`P(w_t | w_{<t}) = softmax(W_y h_t)`。论文报告在 Penn Treebank 上把 perplexity 从 KN5-gram 的 141 降到 112,首次让神经网络在主流语言模型 benchmark 上超过统计模型。这个工作直接催生了 2013 年的 Word2Vec(用 RNN 风格的预测任务学词向量),并把 RNN 推回到 NLP 的舞台中心。

## 训练细节

| 维度 | 简单 RNN 时代的典型做法 |
|------|------|
| 隐状态维度 `d` | 50–500(受当时算力限制,2010 之后才到 1000+) |
| 激活函数 | tanh(隐藏层)、softmax(输出层分类) |
| 初始化 `W_h` | 早期用小高斯;Bengio 1994 后开始关注谱半径,逐步引入正交初始化 |
| 优化器 | SGD + Momentum,梯度爆炸时配合 gradient clipping(`τ = 5`) |
| 截断 BPTT | 长序列上完整 BPTT 计算量爆炸,实际用 **truncated BPTT**:每 `k`(典型 20–50)步截断一次,只回传 k 步内的梯度 |
| 学习率 | `10^-2` ~ `10^-3`,需要随训练衰减;loss spike 比 MLP 频繁 |

## 关键代码

PyTorch 里手写一个最简单的 Elman RNN cell 是这样:

```python
import torch
import torch.nn as nn

class SimpleRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)  # bias 在 W_h 上

    def forward(self, x_t, h_prev):
        # x_t: [B, input_dim], h_prev: [B, hidden_dim]
        h_t = torch.tanh(self.W_x(x_t) + self.W_h(h_prev))
        return h_t

def rnn_forward(cell, x_seq, h_0):
    # x_seq: [B, T, input_dim], h_0: [B, hidden_dim]
    h = h_0
    outputs = []
    for t in range(x_seq.size(1)):
        h = cell(x_seq[:, t, :], h)
        outputs.append(h)
    return torch.stack(outputs, dim=1)  # [B, T, hidden_dim]
```

注意 `for t in range(T)` 这一行——RNN 本质上是**串行的**,第 `t` 步必须等第 `t-1` 步算完。这一点 GPU 无法加速,直接导致循环网络在长序列上训练慢、推理慢,这是 2017 年 Transformer 用 self-attention 把这条循环主轴彻底扔掉的根本动机。

## 影响 / 后续

简单 RNN 在 1986–1996 这十年里是序列建模的唯一通用工具,但实用中的两个问题——**梯度消失/爆炸** 和 **串行不可并行**——决定了它无法独立扛起 NLP 的大规模建模任务。这两件事的解决分别由 LSTM(1997)和 Transformer(2017)接力完成。

但简单 RNN 留下的核心思想——**用一个固定维度的隐状态在时间上递推,共享同一组参数处理变长序列**——是所有后续序列模型的共同祖先。LSTM、GRU 是它的"门控加固版";Seq2Seq 是它的"编码器-解码器封装版";Transformer 抛弃了循环但保留了"序列上下文压进固定维度向量"的瓶颈思想——只不过 Transformer 用 attention 让每个位置同时看到所有其他位置,把"沿时间逐步压缩"换成了"全局并行聚合"。

→ [02-lstm.md](02-lstm.md) · 用门控加上一条细胞状态高速公路,让梯度能稳定走过 100+ 步
→ [04-seq2seq.md](04-seq2seq.md) · 把 RNN 封装成 encoder-decoder,变长输入→变长输出的统一范式
→ [../05-transformer/](../05-transformer/) · 用 self-attention 替代循环,把"串行展开"变成"并行聚合"
→ [../foundations/04-normalization/](../foundations/04-normalization/) · LayerNorm 在 RNN 上的应用是 Transformer 之前稳定深层循环的关键技术
