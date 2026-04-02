# RNN 独立章节设计

## 背景

当前仓库中，RNN/LSTM/GRU 的主体内容位于 `01-Visual-Intelligence/sequence-models/README.md`，但从项目总叙事看，这部分内容更适合作为语言线中 Attention/Transformer 的前置章节，而不是视觉线主章节。

现状存在两个问题：

1. 阶段归属不清。RNN/Seq2Seq 与 Attention、Transformer 的演化关系更贴近 `02-Language-Transformers`。
2. 导航与正文不一致。`01-Visual-Intelligence/README.md` 中“序列模型”简介写的是 GAN/VAE/WaveNet/AlphaGo，但对应正文实际讲的是 RNN 家族。

本设计的目标是在语言线中新增一个独立的 RNN 章节，覆盖 `RNN / LSTM / GRU + Seq2Seq`，并清理最直接的导航冲突。

## 目标

1. 在 `02-Language-Transformers` 下新增独立章节 `recurrent-networks/`。
2. 提供中英双语内容：`README.md` 与 `README_EN.md`。
3. 让该章节成为 Attention/Transformer 的明确前置节点。
4. 修正阶段索引中的直接冲突，避免读者在视觉线和语言线中看到重复且定位矛盾的 RNN 内容。

## 非目标

1. 不在本次改动中重写整个 `01-Visual-Intelligence/sequence-models` 模块。
2. 不在本次改动中新增代码示例仓库、notebook 或测试脚本。
3. 不扩展到 Attention 机制细节推导；Attention 仍由现有章节负责。

## 信息架构

新增目录：

- `02-Language-Transformers/recurrent-networks/README.md`
- `02-Language-Transformers/recurrent-networks/README_EN.md`

配套更新：

- `02-Language-Transformers/README.md`
- `02-Language-Transformers/README_EN.md`（当前缺失，本次补齐）
- `01-Visual-Intelligence/README.md`
- `01-Visual-Intelligence/README_EN.md`（如存在则同步）
- `01-Visual-Intelligence/sequence-models/README.md`
- `01-Visual-Intelligence/sequence-models/README_EN.md`

## 章节定位

新章节标题建议使用“循环神经网络与 Seq2Seq”对应的英文 “Recurrent Networks and Seq2Seq”，目录路径使用 `recurrent-networks/`。

该章节承担的职责：

1. 解释为什么固定窗口方法不足以处理序列依赖。
2. 解释基本 RNN 的状态传递和训练困难。
3. 解释 LSTM/GRU 如何通过门控缓解长期依赖问题。
4. 解释 Seq2Seq 如何把循环网络用于序列到序列生成任务。
5. 为 Attention 章节建立明确动机，但不重复 Attention 的主体内容。

## 章节结构

建议按以下结构组织中英文正文：

1. 为什么序列建模需要循环结构
2. 基本 RNN：状态传递、时间展开与 BPTT
3. 梯度消失与梯度爆炸：为什么长依赖难学
4. LSTM：显式记忆通道与门控机制
5. GRU：更轻量的门控设计
6. Seq2Seq：编码器-解码器如何处理输入输出序列
7. 工程实践：变长序列、padding、packed sequence、梯度裁剪
8. 为什么 Seq2Seq 仍有信息瓶颈，并自然过渡到 Attention

每节控制在“概念解释 + 关键公式/图示替代文本 + 工程直觉”的粒度，避免把章节做成论文式推导。

## 内容边界

本章包含：

1. `RNN / LSTM / GRU` 的基本机制与直觉解释。
2. 至少一个 `LSTM vs GRU` 对比表。
3. 至少一个最小 PyTorch 代码示例，覆盖循环层 API 或变长序列处理。
4. `Seq2Seq` 的基础结构与其信息瓶颈。
5. 与 Attention 章节的前后跳转链接。

本章不包含：

1. Bahdanau / Luong Attention 的完整推导。
2. Transformer 编码器/解码器细节。
3. 大规模预训练模型历史。

## 导航与链接策略

### Phase 02

`02-Language-Transformers/README.md` 需要将新章节插入到 Attention 之前。建议顺序为：

1. 循环神经网络与 Seq2Seq
2. 注意力机制
3. Transformer 架构
4. 预训练模型

如果补齐 `02-Language-Transformers/README_EN.md`，其中顺序必须与中文保持一致。

### Phase 01

`01-Visual-Intelligence/README.md` 当前“序列模型”简介与正文错位。本次不重做其主题，但要消除最明显冲突。推荐保守处理：

1. 把视觉线中的该条目改写为过渡性说明，明确“RNN 主体已迁移到语言线”。
2. 在 `01-Visual-Intelligence/sequence-models/README.md` 顶部增加说明或迁移提示，指向新的语言线章节。

这样可以避免本次任务演变成对 Phase 01 的完整重写。

## 文案风格

遵循仓库现有中文技术文风：

1. 先讲“为什么出现”，再讲“如何工作”。
2. 公式适量，优先解释设计直觉。
3. 章节末尾给出“你要记住”的总结式句子，强化演化逻辑。
4. 工程建议聚焦最常见坑点，不追求面面俱到。

英文版不要求逐字直译，但要求结构、信息密度和导航一致。

## 实施范围

### 必做

1. 新增 `recurrent-networks` 中英双语章节。
2. 更新 `02-Language-Transformers` 阶段索引。
3. 处理 `01-Visual-Intelligence` 中与 RNN 迁移直接相关的导航冲突。
4. 检查前后跳转链接，避免死链。

### 暂不做

1. 重写视觉线“序列模型”为 GAN/VAE/WaveNet 新正文。
2. 为新章节补充独立示意图或 notebook。
3. 调整顶层 `README.md` 的模块描述，除非实现时发现必须同步。

## 风险与取舍

### 风险 1：Phase 01 中仍残留历史命名负担

即使加入迁移提示，`sequence-models` 这个名字在视觉线中仍显得含混。

取舍：本次只处理直接冲突，不做全量重构，保持任务边界稳定。

### 风险 2：双语一致性

当前 `02-Language-Transformers` 目录缺少 `README_EN.md`，而其他章节通常有双语索引。

取舍：本次应补齐英文索引，否则新英文章节会缺少阶段入口。

### 风险 3：与 Attention 章节重复

如果在 RNN 章节中过度展开 Seq2Seq 信息瓶颈与 Attention，内容会与后续章节重叠。

取舍：RNN 章节只讲“为什么需要 Attention”，不讲 Attention 内部机制。

## 验证标准

完成后应满足以下条件：

1. `02-Language-Transformers/recurrent-networks/` 中英文件存在且内容结构对齐。
2. `02-Language-Transformers` 阶段索引可导航到新章节。
3. 新章节可顺畅跳转到 Attention 或 Transformer 相关后续章节。
4. `01-Visual-Intelligence` 不再把 RNN 当作视觉线主章节内容来介绍。
5. 仓库内不存在新增的错误相对链接。

## 实现建议

实现时按以下顺序进行：

1. 创建新章节中英文文稿。
2. 更新 Phase 02 索引与前后跳转。
3. 处理 Phase 01 迁移提示与描述修正。
4. 通读检查中英文导航和链接一致性。

## 开放问题

当前无阻塞性开放问题。唯一需要在实现时现场确认的是 `01-Visual-Intelligence/README_EN.md` 是否存在；若不存在，则只修改存在的英文入口文件。
