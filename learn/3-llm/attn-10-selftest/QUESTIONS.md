# 注意力系列自测题库

单选题，共 27 题。点开“答案”查看。也可以用[网页版](demo/index.html)随机出题。

**1. [01 基础]** 一层单头注意力里，n 个词的分数表形状是？

- A. [n, d]
- B. [n, n]
- C. [d, d]
- D. [n, 1]

<details><summary>答案</summary>

B。每个词给每个词打分，所以是 n×n。

</details>

**2. [01 基础]** 对分数表每一行做 softmax 后，每行的和是？

- A. 0
- B. 1
- C. n
- D. d

<details><summary>答案</summary>

B。softmax 输出是和为 1 的正数。

</details>

**3. [01 基础]** 上下文长度变成原来的 10 倍，分数表的元素个数变成？

- A. 10 倍
- B. 20 倍
- C. 100 倍
- D. 1000 倍

<details><summary>答案</summary>

C。n×n：(10n)² = 100·n²。

</details>

**4. [01 基础]** 为什么用 W_Q 和 W_K 两个不同矩阵，而不是直接拿 x 和 x 比？

- A. 为了减少参数
- B. “我想找什么”和“我能被怎样找到”是两件事，分开学更灵活，打分也不必对称
- C. 因为 x 太长
- D. 因为 softmax 要求

<details><summary>答案</summary>

B。q 与 k 是同一个词的两种角色，用不同矩阵投影。

</details>

**5. [03 掩码]** 因果掩码把未来位置的分数设为？

- A. 0
- B. 1
- C. −∞
- D. +∞

<details><summary>答案</summary>

C。softmax 里 e^−∞ = 0，未来位置权重恰好为 0。

</details>

**6. [03 位置]** 没有位置编码时，打乱词序，每对词之间的注意力权重？

- A. 完全不变（只是行列重排）
- B. 全部变为 0
- C. 变成均匀
- D. 随机变化

<details><summary>答案</summary>

A。注意力对输入顺序是等变的，本身看不见词序。

</details>

**7. [02 手算]** X=[[1,0],[0,1],[1,1]]，W_Q=[[1,0],[1,1]]。Q 的第三行是？

- A. [1,1]
- B. [2,1]
- C. [1,2]
- D. [2,2]

<details><summary>答案</summary>

B。[1,1]·W_Q = [1×1+1×1, 1×0+1×1] = [2,1]。

</details>

**8. [03 形状]** B=2, n=10, d_model=64, h=8。分数 S 的形状是？

- A. [2, 10, 64]
- B. [2, 8, 10, 10]
- C. [2, 8, 10, 8]
- D. [2, 10, 10]

<details><summary>答案</summary>

B。每个头一张 n×n 表：[B, h, n, n]。

</details>

**9. [03 形状]** d_model=512, h=8，每个头的 d_k 是？

- A. 8
- B. 64
- C. 512
- D. 4096

<details><summary>答案</summary>

B。d_k = d_model / h = 64。

</details>

**10. [03 形状]** 不计偏置，d_model=512 的一层多头注意力有多少参数？

- A. 262,144
- B. 524,288
- C. 786,432
- D. 1,048,576

<details><summary>答案</summary>

D。W_Q、W_K、W_V、W_O 各 512×512，共 4×512² = 1,048,576。

</details>

**11. [04 softmax]** softmax([0, 0]) 等于？

- A. [0, 0]
- B. [1, 1]
- C. [0.5, 0.5]
- D. [0, 1]

<details><summary>答案</summary>

C。两个相等的分数得到相等的权重，和为 1。

</details>

**12. [04 softmax]** 把所有分数同时加上 5，softmax 的结果？

- A. 变大
- B. 变小
- C. 不变
- D. 变成 one-hot

<details><summary>答案</summary>

C。平移不变性：分子分母同乘 e⁵。这是“先减最大值”的原理。

</details>

**13. [04 softmax]** q、k 各分量独立、均值 0、方差 1，d_k=64。q·k 的标准差是？

- A. 1
- B. 8
- C. 64
- D. 4096

<details><summary>答案</summary>

B。方差为 d_k = 64，标准差 √64 = 8。

</details>

**14. [04 softmax]** 除以 √d_k 的主要目的是？

- A. 让分数尺度与 d_k 无关，避免 softmax 饱和导致梯度消失
- B. 让矩阵乘法更快
- C. 让权重和为 1
- D. 减少显存

<details><summary>答案</summary>

A。Var(q·k/√d_k)=1。

</details>

**15. [04 softmax]** softmax 输出接近 one-hot 时，它的梯度（Jacobian）？

- A. 很大
- B. 接近 0
- C. 等于 1
- D. 无定义

<details><summary>答案</summary>

B。J = diag(a) − aaᵀ，a 接近 one-hot 时每一项都趋于 0。

</details>

**16. [06 位置]** RoPE 把旋转作用在哪些向量上？

- A. q 和 k
- B. 只在 v
- C. q、k、v 全部
- D. 词向量加法

<details><summary>答案</summary>

A。位置只需影响“谁和谁匹配”，所以只旋转 q 和 k。

</details>

**17. [06 位置]** RoPE 下，q 与 k 旋转后的点积取决于？

- A. 两者的绝对位置
- B. 两者位置之差 m−n
- C. 序列长度
- D. 头的编号

<details><summary>答案</summary>

B。R(mθ)ᵀR(nθ)=R((n−m)θ)。

</details>

**18. [06 位置]** 8 个头的 ALiBi，斜率最大（1/2）的头？

- A. 几乎只看近处
- B. 能看到很远
- C. 不使用偏置
- D. 只看第一个词

<details><summary>答案</summary>

A。每远一格扣 0.5 分，权重迅速集中到近处。

</details>

**19. [06 位置]** 原始 Transformer 的正弦位置编码怎么用？

- A. 旋转 q、k
- B. 加到输入的词向量上
- C. 加在分数上
- D. 拼接到 v

<details><summary>答案</summary>

B。x ← x + PE(pos)。

</details>

**20. [05 多头]** 拼接各头输出再乘 W_O，等价于？

- A. 各头输出分别乘 W_O 对应的行块，再求和
- B. 只取第一个头
- C. 对各头取平均
- D. 对各头取最大

<details><summary>答案</summary>

A。分块矩阵乘法。

</details>

**21. [05 多头]** 取 d_k = d/h 的好处是？

- A. 多头与单头 d_k=d 参数量基本相同，换来多套匹配规则
- B. 头越多越快
- C. 不需要 W_O
- D. softmax 不再需要缩放

<details><summary>答案</summary>

A。投影参数同为 4d²。

</details>

**22. [09 高效]** L=32，8 个 KV 头，d_head=128，fp16。每个词的 KV cache 是？

- A. 16 KiB
- B. 64 KiB
- C. 128 KiB
- D. 1 MiB

<details><summary>答案</summary>

C。2×32×8×128×2 字节 = 131,072 字节 = 128 KiB。

</details>

**23. [09 高效]** 64 个 q 头、8 组 KV 的 GQA，相比 MHA，KV cache 缩小多少？

- A. 2 倍
- B. 8 倍
- C. 64 倍
- D. 不变

<details><summary>答案</summary>

B。缩小倍数 = h/g = 64/8 = 8。

</details>

**24. [09 高效]** FlashAttention 与标准注意力的输出相比？

- A. 是近似，略有误差
- B. 数学上精确相同，只是计算顺序不同
- C. 只在 n 很大时相同
- D. 没有可比性

<details><summary>答案</summary>

B。它是精确注意力，快在减少对慢速显存的读写。

</details>

**25. [09 高效]** 在线 softmax 里，新块最大值比旧的 m 小，旧累计量要乘的缩放因子是？

- A. 0
- B. 1
- C. −1
- D. e

<details><summary>答案</summary>

B。m′=m，因子 e^(m−m′)=e⁰=1。

</details>

**26. [00 动机]** RNN 与自注意力相比，最大路径长度分别是？

- A. RNN O(1)，注意力 O(n)
- B. RNN O(n)，注意力 O(1)
- C. 都是 O(n)
- D. 都是 O(1)

<details><summary>答案</summary>

B。Vaswani 2017 Table 1：循环 O(n)，自注意力 O(1)。

</details>

**27. [00 动机]** n=64, d=512。比较 n²d 与 nd²，哪个大？

- A. n²d 大 8 倍
- B. nd² 大 8 倍
- C. 相等
- D. n²d 大 64 倍

<details><summary>答案</summary>

B。n²d=2,097,152；nd²=16,777,216，nd² 大 8 倍（d/n=8）。

</details>

