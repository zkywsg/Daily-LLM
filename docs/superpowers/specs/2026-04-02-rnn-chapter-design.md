# RNN 独立章节 Spec

## 背景

当前仓库中，RNN/LSTM/GRU 的主体内容位于 `01-Visual-Intelligence/sequence-models/README.md`，但从项目叙事看，这部分内容更适合作为 `02-Language-Transformers` 中 Attention / Transformer 的前置章节。

这次工作的目标不是抽象地“新增一个目录”，而是按项目既有 README 规范，为语言线补一篇独立的 RNN 章节，并同步检查时间线与模块索引的一致性。

本 spec 以 `CLAUDE.md` 和 `STYLE.md` 为准，重点约束：

1. 新章节 README 必须符合模块模板。
2. 新章节必须带上项目签名元素。
3. 模块新增后必须检查 `00-Timeline/README.md` 的联动更新。

## 目标

1. 在 `02-Language-Transformers` 下新增独立章节 `recurrent-networks/`。
2. 章节内容覆盖 `RNN / LSTM / GRU + Seq2Seq`，作为 Attention 的直接前置。
3. 新章节遵循 `STYLE.md` 定义的 README 结构与写作风格。
4. 同步修正时间线中与 RNN 历史归属直接相关的链接或模块指向。

## 非目标

1. 不在本次工作中重写整个 `01-Visual-Intelligence/sequence-models` 正文。
2. 不在本次工作中扩展 Attention 机制细节或 Transformer 主体内容。
3. 不在本次工作中新增 notebook、`src/` 代码文件或图像资源。

## 文件范围

### 必做

- `02-Language-Transformers/recurrent-networks/README.md`
- `02-Language-Transformers/recurrent-networks/README_EN.md`
- `02-Language-Transformers/README.md`
- `00-Timeline/README.md`

### 按实际存在或必要性决定

- `02-Language-Transformers/README_EN.md`
- `01-Visual-Intelligence/README.md`
- `01-Visual-Intelligence/sequence-models/README.md`
- `01-Visual-Intelligence/sequence-models/README_EN.md`

## README 规范约束

新章节最终 README 必须遵循 `STYLE.md` 规定的模块结构，而不是自由发挥。

### 标题

标题格式必须是：

`# 为什么 [旧方法] 不够用了？—— [技术名]`

本章建议标题：

`# 为什么固定窗口不够用了？—— 循环神经网络与 Seq2Seq`

英文版标题应保持同样的“旧方法不够用 -> 新技术出现”的结构，而不是只写技术名。

### 这个问题从哪来

这一节必须放在模块开头，并满足以下要求：

1. 引用年份与代表工作，而不是只写笼统背景。
2. 交代固定窗口或浅层序列方法为何难以处理长距离依赖。
3. 引出 RNN 的状态传递、LSTM/GRU 的优化动机，以及 Seq2Seq 的任务扩展。

建议覆盖的历史节点：

1. 1997 `LSTM`
2. 2014 `Seq2Seq`
3. 2014 `GRU`

如果文案需要再补充一条早期 RNN 背景，可在实现时加入，但不要求在 spec 中继续扩展范围。

### 学习目标

必须使用“完成后你应能回答”这一固定形式，并落成 3 个具体问题。建议目标如下：

1. RNN 为什么比固定窗口方法更适合序列建模？
2. LSTM / GRU 缓解了什么问题，代价是什么？
3. Seq2Seq 为什么有效，又为什么最终逼出了 Attention？

### 正文结构

正文必须保持 `直觉 -> 机制 -> 工程陷阱` 三段式。

#### 1. 直觉

限制在生活类比和问题直觉层面，避免一上来堆公式。应覆盖：

1. 序列任务依赖上下文。
2. RNN 的核心直觉是“把前文压进持续更新的隐藏状态”。
3. Seq2Seq 的核心直觉是“先编码，再生成”，但固定长度状态会形成信息瓶颈。

#### 2. 机制

必须按 `公式 -> Mermaid -> 渐进式实现` 组织。

建议机制部分按以下顺序展开：

1. 基本 RNN 更新公式与时间展开
2. BPTT 与梯度消失/爆炸
3. LSTM 的记忆通道与门控
4. GRU 的轻量门控结构
5. Seq2Seq 编码器-解码器
6. 最小 PyTorch 渐进式实现

渐进式实现必须体现“每一步解决什么问题”，而不是机械堆功能。建议最少包含：

1. Step 1：最小 RNN / LSTM 前向逻辑
2. Step 2：变长序列、padding 或 `pack_padded_sequence`
3. Step 3：工程完善，如梯度裁剪、双向编码器或输出选择

如果插入 Mermaid 图，需遵循 `STYLE.md` 中的暖色系规范和 `graph TD` 约束。

#### 3. 工程陷阱

必须使用“原因 -> 现象”的写法，并按优先级排序。建议覆盖：

1. 长序列训练不稳定 -> 梯度爆炸、loss 抖动，优先使用梯度裁剪
2. padding / length 处理错误 -> 最后有效时间步取错，分类结果失真
3. 误把 LSTM / GRU 当成万能长依赖解法 -> 长输入仍有压缩瓶颈
4. 低估串行计算成本 -> 训练吞吐成为结构性上限

### 签名元素

本章必须包含 `STYLE.md` 定义的三个签名元素：

1. `这个问题从哪来`
2. `你要记住`
3. `演进笔记`

其中 `你要记住` 全章不超过 3 次，优先放在以下位置：

1. 解释 RNN 的真正瓶颈之后
2. 解释 LSTM / GRU 的价值之后
3. 解释 Seq2Seq 到 Attention 的过渡之后

### 演进笔记

模块结尾必须有 `演进笔记`，并使用“解决了什么，遗留了什么新问题”的句式。

本章应明确写出：

1. RNN 解决了序列可以端到端建模的问题。
2. LSTM / GRU 缓解了长依赖难训的问题。
3. Seq2Seq 让生成式序列任务成立。
4. 固定长度上下文和串行计算仍是硬限制，因此自然推进到 Attention。

结尾链接应指向 Attention 章节，而不是直接跳到 Transformer。

### 上一章 / 下一章

本章需要在 spec 中提前约束导航逻辑：

1. **下一章** 应为 `attention-mechanisms/README.md`
2. **上一章** 应优先选择 `02-Language-Transformers/README.md` 的阶段入口，避免从视觉线直接跳转导致路径混乱

英文版导航必须与中文版对应。

## 内容边界

本章包含：

1. RNN 基本机制
2. LSTM / GRU 的核心直觉与关键公式
3. Seq2Seq 的编码器-解码器结构
4. 至少一个最小 PyTorch 示例
5. 从 Seq2Seq 过渡到 Attention 的动机

本章不包含：

1. Bahdanau / Luong Attention 详细推导
2. Self-Attention 与 Multi-Head Attention 主体内容
3. Transformer 内部结构拆解
4. 预训练模型历史

## 时间线联动要求

根据 `CLAUDE.md` 的“时间线与模块双向同步”规则，新增本章后必须检查 `00-Timeline/README.md` 中相关年份的条目归属是否需要同步。

### 重点检查年份

#### 2013

当前有 `LSTM 序列生成 | [01·视觉线]`。本次实现需要评估是否应改为指向语言线的新 RNN 章节，或至少改成不与新章节定位冲突的表述。

#### 2014

当前已有 `Seq2Seq`、`Attention`、`GRU`。本次实现需要明确：

1. `Seq2Seq` 应落到新 RNN 章节
2. `GRU` 应从视觉线归属调整为语言线
3. `Attention` 保持在现有注意力章节

#### 2015

当前有 `char-rnn / 语言生成`。本次实现需要检查它是否应链接到新 RNN 章节。

#### 2017

`Transformer：把 RNN 彻底扔掉` 是本章的自然后继。实现时需要确认从新章节到 Attention，再到 Transformer 的链接链路清晰且不打架。

## 阶段索引要求

### Phase 02

`02-Language-Transformers/README.md` 需要把新章节插入到 Attention 之前。建议顺序：

1. 循环神经网络与 Seq2Seq
2. 注意力机制
3. Transformer 架构
4. 预训练模型

如果新增或补齐 `02-Language-Transformers/README_EN.md`，顺序必须同步。

### Phase 01

本次不重写 `01-Visual-Intelligence/sequence-models` 的正文，但如果新章节上线后出现明显导航冲突，可以做最小修正：

1. 在视觉线索引中移除对 RNN 的主入口暗示
2. 或在旧页面顶部增加迁移提示，指向新的语言线章节

此处只做最小一致性修复，不扩展为整章重构。

## 文风约束

实现时应继续遵循项目现有中文技术文风：

1. 先讲“为什么出现”，再讲“如何工作”
2. 公式适量，不写成教科书式长推导
3. 工程部分聚焦最常踩坑点
4. 章节末尾通过 `演进笔记` 明确技术遗产与下一步动机

英文版不要求逐字直译，但必须保持：

1. 结构一致
2. 导航一致
3. 核心知识点一致

## 验收标准

完成后应满足：

1. 新增的 RNN 章节 README 符合 `STYLE.md` 模板
2. 三个签名元素完整出现
3. 模块结尾存在正确的 `演进笔记` 和下一章链接
4. `02-Language-Transformers/README.md` 能导航到新章节
5. `00-Timeline/README.md` 中与 RNN / GRU / Seq2Seq 相关的模块归属不再与新章节定位冲突
6. 中英文版本结构一致，若英文阶段索引存在则入口一致

## 实现顺序建议

1. 先写中文 README，确保结构完全符合模板
2. 再写英文 README，保持结构与导航对应
3. 更新 `02-Language-Transformers/README.md`
4. 检查并同步 `00-Timeline/README.md`
5. 最后只对必要的旧入口做最小修正
