# 注意力 10：综合自测与小项目

**用法**：学完前 9 篇后做。题库在 [QUESTIONS.md](QUESTIONS.md)（GitHub 上直接读，答案折叠）；也可以用[网页版](demo/index.html)：随机出题、选完即给解释、最后汇总错题。

**配套**：[网页自测](demo/index.html) · [题库 QUESTIONS.md](QUESTIONS.md) · [TikTok 图卡](cards/out/)。

## 题库覆盖

27 道单选题：

| 篇 | 题数 | 考什么 |
|---|---|---|
| 00 动机 | 2 | 路径长度、n²d 与 nd² 的比较 |
| 01 基础 | 4 | 分数表形状、softmax 行和、n² 增长、为什么两个矩阵 |
| 02 手算 | 1 | 矩阵乘法 |
| 03 掩码、位置、形状 | 5 | −∞、位置不可见、形状、参数量 |
| 04 softmax | 5 | 平移不变、方差、√d 的目的、梯度 |
| 05 多头 | 2 | 求和等价、d_k=d/h |
| 06 位置 | 4 | RoPE 对象、相对性、ALiBi、正弦 |
| 09 高效 | 4 | KV cache 大小、GQA、FlashAttention、在线 softmax |

可计算的题（KV cache、方差、参数量、手算矩阵、n²d 比较）答案已用脚本独立核对。概念题的答案依据各篇的推导与引用文献。

## 五个小项目

按难度排序。每个都对应前面某一篇，能做出来就说明那一篇真吃透了。

1. **验证 RoPE 的相对性**（注意力 06）：用 NumPy 实现 `rope(x, pos)`，随机 q、k，验证 `rope(q, m) @ rope(k, n)` 只依赖 `m-n`。
2. **给从零实现加上滑动窗口**（注意力 09 + [注意力 07 的代码](../attn-07-from-scratch/code/attention.py)）：写一个 `window_mask(n, w)`，接进 `scaled_dot_product_attention`，写测试：窗口外的权重为 0。
3. **实现分块在线 softmax 注意力**（注意力 09）：不一次性算 $n\times n$，按 K、V 块遍历，与普通注意力对比最大误差（参考 [`online_softmax.py`](../attn-09-efficient/code/online_softmax.py) 的单行版，扩展到整个矩阵）。
4. **给从零实现加 GQA**（第五、九篇）：让 K、V 只有 $g$ 个头，每个头服务 $h/g$ 个 q 头；测试 $g=h$ 时与原实现一致，$g=1$ 时就是 MQA。
5. **在 GPT-2 里找 previous-token 头**（注意力 08）：运行提取脚本，写统计找出“对角线下一格”权重最高的头，并画出来。（需要先安装 torch 与 transformers，此项我未运行过。）

## 下一步

注意力系列到此结束。接着可以进入完整 Transformer 块（残差、LayerNorm、前馈网络），再到预训练与对齐。这些在[学习路线](../../README.md)里还没有写。
