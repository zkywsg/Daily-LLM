# 注意力 09：高效注意力，KV cache、GQA、FlashAttention

**读完你会知道**：生成时为什么要缓存 K 和 V、缓存有多大、MQA/GQA 怎么把它变小、FlashAttention 为什么不改变结果却更快、滑动窗口怎么降低 $n^2$。

**前置**：[形状](../attn-03-shapes/)、[softmax 与 √d](../attn-04-softmax-scale/)。

**配套**：[互动版](demo/index.html)（KV cache 计算器、生成步骤、在线 softmax 步进、滑动窗口）· [验证脚本](code/online_softmax.py)（NumPy，已运行）· [TikTok 图卡](cards/out/)。

## 第 1 层 · 一句话

生成文字时，旧词的 K、V 不会变，缓存起来只算新词那一行；缓存太大就让多个查询头共用少数 K/V 头（MQA/GQA）；分数表太大，就分块算、边算边归并（FlashAttention）；或者只看一个窗口。

## 第 2 层 · 白话

- **KV cache**：写作文每写一个字都要回头看前文。前文每个字的“键”和“值”写完就不变，存着就行，不用每次重算。代价是占内存，随长度线性增长。
- **MQA / GQA**：查询（q）的头保持很多，但键值（K、V）的头只留 1 个或几个，让一组 q 头共享同一份 K/V。缓存按比例缩小。
- **FlashAttention**：结果与普通注意力完全一致，只改变计算顺序：把分数表切成小块，每块在高速缓存里算完就合并进结果，不把整张 $n\times n$ 表写回慢速显存。
- **滑动窗口**：每个词只看最近 $w$ 个词，分数表从 $n^2$ 变成 $n\cdot w$。

## 第 3 层 · 手算

**KV cache 大小**（每个词、每层存 K 和 V，各 $n_{kv}$ 个头 × $d_{head}$ 维，fp16 每数 2 字节）：

$$
\text{每词字节}=2\,(K,V)\times L\times n_{kv}\times d_{head}\times 2
$$

| 配置（公开资料，需自行核对） | 每词 | 序列长度 | 总计（1 个序列） |
|---|---|---|---|
| GPT-3 175B：$L=96$，96 个头，$d_{head}=128$（MHA） | 4.5 MiB | 2048 | 9 GiB |
| LLaMA-2 70B：$L=80$，64 个 q 头、8 个 KV 头，$d_{head}=128$（GQA） | 320 KiB | 4096 | 1.25 GiB |
| 同上改成 MHA（64 个 KV 头）的假想对照 | 2.5 MiB | 4096 | 10 GiB |

GQA 把 KV 头从 64 减到 8，缓存缩小 8 倍。

**有无缓存的计算次数**：生成 $n=100$ 个词，每步每个头：

| | 第 $t$ 步的 $q\cdot k$ 次数 | 总计 |
|---|---|---|
| 有缓存 | $t$（新 q 对 $t$ 个 k） | 5,050 |
| 无缓存 | $t^2$（整张表重算） | 338,350（约 67 倍） |

（只比较分数的点积次数，不含投影与其他开销。）

## 第 4 层 · 数学推导

**(1) 在线 softmax**

目标：对一行分数 $s_1,\dots,s_n$ 和值 $v_1,\dots,v_n$ 计算 $o=\sum_j \frac{e^{s_j}}{\sum_k e^{s_k}}v_j$，但分数分块到来。维护三个量：当前最大值 $m$，指数和 $\ell$，加权累加 $o$。读入一块（最大值 $m_b$）后：

$$
m'=\max(m,m_b),\quad \ell'=\ell\,e^{m-m'}+\sum_{j\in b}e^{s_j-m'},\quad o'=o\,e^{m-m'}+\sum_{j\in b}e^{s_j-m'}v_j
$$

全部读完后 $o/\ell$ 就是精确的 softmax 加权结果。$e^{m-m'}$ 把旧的累计量换算到新的参照最大值下，这是平移不变性（见注意力 04）的直接应用。

**(2) FlashAttention 的思想**

Dao et al. 2022：以上技巧加上分块（tiling），让每个 $Q$ 块依次与各个 $K,V$ 块计算，只在片上高速存储中保留小块，不把完整的 $n\times n$ 分数表写入显存（HBM）。它是**精确**注意力，不是近似；加速来自减少对慢速显存的读写（IO-aware），而不是减少浮点运算量。具体加速倍数依硬件、序列长度而异，这里不给数字。

**(3) MQA / GQA**

- MQA（Shazeer 2019）：所有 q 头共享 1 组 K/V。
- GQA（Ainslie et al. 2023）：把 $h$ 个 q 头分成 $g$ 组，每组共享 1 组 K/V，$1<g<h$，在质量与缓存之间折中；$g=1$ 即 MQA，$g=h$ 即 MHA。
- 缓存缩小倍数 $=h/g$。

**(4) 滑动窗口**

位置 $i$ 只看 $[i-w+1,\,i]$：分数从 $n^2$ 个降到约 $n\cdot w$ 个。信息可以通过堆叠多层间接传得更远（每层多传 $w$）。

## 第 5 层 · 代码与练习

核心：`code/online_softmax.py`。在 64 个随机分数上，块大小 1、4、16、64 时，在线结果与普通 softmax 的最大差为 $2\text{--}4\times10^{-16}$。

**练习**

1. 某模型 $L=32$，8 个 KV 头，$d_{head}=128$，fp16。每词 KV cache 多大？4096 词呢？
2. 如果改成 fp8（1 字节），缓存怎么变？
3. 在线 softmax 里，若新块的最大值比旧的 $m$ 小，$e^{m-m'}$ 等于多少？
4. 窗口 $w=4$，4 层堆叠，最远能间接看到多远？

**答案要点**

1. $2\times32\times8\times128\times2=131{,}072$ 字节 = 128 KiB/词；4096 词 = 512 MiB。
2. 减半。
3. $m'=m$，因子为 $e^0=1$，旧累计量不变。
4. 每层多传 $w-1=3$ 个位置的信息，粗略约 $4\times3=12$ 个位置之外（精确边界取决于是否包含自身，这里只给量级）。

## 来源

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019（MQA）。
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023。
- Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018。
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022。
- 模型配置已核对：GPT-3 175B 为 96 层、96 头、d_model=12288（头维 128）、上下文 2048（见 OPT 论文的架构表与 Lambda 的技术概述）；LLaMA-2 70B 为 80 层、64 个 q 头、8 个 KV 头、hidden 8192（头维 8192/64=128；见 LLaMA-2 论文与 HF 配置）。字节计算为本仓库原创，已用脚本核对。
