# 注意力 08：真实模型里的注意力长什么样

**状态说明**：本篇的脚本和页面**没有在真实模型上运行过**。本仓库没有安装 torch / transformers，也没有下载 GPT-2，所以这里没有任何来自真实模型的数字或截图。页面自带的示例是[玩具模型](../_shared/engine.js)数据，页面上已标注。你运行脚本后，页面可以直接加载真实结果。

**配套**：[提取脚本](code/extract_attention.py)（已过语法检查，未真实运行）· [可视化页](demo/index.html)（可粘贴 JSON）· [TikTok 图卡](cards/out/)。

## 第 1 层 · 一句话

把真实模型的注意力权重取出来画成热力图，你会看到不同的头各有各的“习惯”，有些整齐，有些看不出规律。

## 第 2 层 · 白话

前面所有例子的矩阵都是手设的，真模型的矩阵是训练出来的。你看到的“头 5 层 3”是一张 $n\times n$ 的权重表，和之前的点网格完全同类，只是更多、更杂。读它的方法也一样：每行是一个提问的词，每列是被看的词。

## 第 3 层 · 手算（怎么读）

对 GPT-2 small 来说，一句 $n$ 个 token 的话会得到 $12$ 层 × $12$ 个头 = 144 张 $n\times n$ 的表（层数与头数来自模型配置，脚本里有形状断言）。因为是因果模型，每张表的右上三角都是 0。

读图时先看三件事：
1. 亮点是不是集中在**对角线下面一格**（看前一个词）。
2. 是不是大部分行都亮着**第一个 token**。
3. 是不是“上一次出现同一个词之后”的位置亮（重复模式）。

## 第 4 层 · 文献里提到过的几种头

以下是文献中的描述，**我没有在本仓库里复现**，请把它们当作“该找什么”的线索，而不是结论：

| 现象 | 描述 | 出处 |
|---|---|---|
| 上一个词的头（previous-token head） | 每个位置主要看前一个位置 | Elhage et al. 2021 等 |
| 归纳头（induction head） | 遇到 `A B … A` 时，倾向于在第二个 `A` 处关注上一次 `A` 后面的 `B`，帮助重复和模式补全 | Olsson et al. 2022 |
| 注意力沉降（attention sink） | 大量权重落在序列最开头的几个 token 上 | Xiao et al. 2023（StreamingLLM） |

不同模型、不同层的表现会很不一样；“头有可解释功能”也不是对所有头成立。

## 第 5 层 · 代码与练习

```bash
pip install torch transformers
python code/extract_attention.py "The cat did not cross the road because it was too tired" > attn.json
```

然后打开 `demo/index.html`，把 `attn.json` 的内容粘贴进去（或点“选择文件”）。

**练习（运行后做）**

1. 遍历 12 层 12 头，统计每个头“对角线下一格”的平均权重，找出最像 previous-token head 的头。
2. 用 `A B C D A B C D A B` 这样重复的 token 序列，找出在第二、三轮里“看上一轮下一个词”的头（归纳头候选）。
3. 统计每个头分配给第一个 token 的平均权重，看哪几层最高。
4. 对同一句话换大小写、换一个同义词，看某个头的图变化多大。

这些练习的答案取决于真实运行结果，这里不预设。

## 来源

- Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021。
- Olsson et al., *In-context Learning and Induction Heads*, 2022。
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023。
- 归纳头的 `[A][B]…[A]→[B]` 模式、StreamingLLM 的“开头 token 成为注意力沉降”两点已对照二手介绍核对，原论文细节请自行查阅；也有后续工作指出沉降的作用并不总是正面。脚本和页面为本仓库原创。
