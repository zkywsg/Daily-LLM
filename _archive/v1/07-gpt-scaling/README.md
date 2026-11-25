# 大语言模型(GPT 系 + Scaling)

> **把语言模型当基础设施:用无监督生成预训练 + 规模化扩展,让一个模型不微调就能做几乎所有 NLP 任务。**

## 一句话定位

这家族解决的是 NLP 这件事里一个最根本的范式问题——**模型应该为每个任务单独训练,还是先学一个通用的语言能力再适配下游?** 2017 年之前的 NLP 默认是前者:翻译用 Seq2Seq、分类用 CNN-Text、NER 用 BiLSTM-CRF,每个任务从头训。2018 年 OpenAI 的 GPT-1 给出了完全不同的方案——**先在大量无标注文本上做自回归预训练学到通用语言表征,再用少量任务数据微调**。这一思路在同年的 [BERT](../06-bert-family/) 上得到了双向版本的验证(GLUE 全面 SOTA),但真正展现"语言模型不只是 NLP 工具、而是某种通用智能基底"的是后续的 scaling 实验:**GPT-2 把模型推到 1.5B 后发现 zero-shot 任务能力开始涌现,GPT-3 推到 175B 后 in-context learning 让"几乎所有任务一个模型搞定"成为现实**。围绕这条规模化轨迹,Kaplan 2020 / Chinchilla 2022 给出了 scaling law 的定量表述,把"加规模 = 加性能"从经验观察推到了可预测的物理定律级别。这家族要回答的是:**从 2018 GPT-1 到 2023 GPT-4 / LLaMA 时代,大语言模型这条线是怎么一步步走出来的**。

## 概念本身

GPT 系的核心是 **decoder-only Transformer + 自回归语言建模 + 规模化**。

**架构选择**——和 [BERT](../06-bert-family/) 的 encoder-only + 双向 masked LM 不同,GPT 系用 [Transformer](../05-transformer/01-transformer.md) 的 **decoder 部分**(去掉 cross-attention),输入序列从左到右单向自回归预测下一个 token:

$$
p(x_1, x_2, \ldots, x_N) = \prod_{t=1}^{N} p(x_t | x_{<t})
$$

模型架构上的差异只是 attention mask(decoder 用 causal mask),其他完全是 [Transformer](../05-transformer/01-transformer.md) 的同款 block。这一选择当时看是有争议的——BERT 双向编码在阅读理解类任务上明显占优,GPT 单向看似自缚。但 GPT 团队赌的是**生成式能力的通用性**:能生成下一个词的模型,本质上学到了完整的语言分布;有了完整分布,任何任务都可以表述成"给定上下文生成答案"。

**Scaling 三轴**——GPT 系的演化主要靠把三个维度一起扩大:

- **参数量 N**:GPT-1 117M → GPT-2 1.5B → GPT-3 175B → GPT-4 ~1.8T(估计)
- **数据量 D**:GPT-1 BookCorpus 800M token → GPT-2 WebText 40B → GPT-3 CommonCrawl 300B → GPT-4 ~13T
- **算力 C**:三者之积近似 `C ≈ 6 × N × D`,这是训练 FLOPs 的标准估计

Kaplan 2020 *Scaling Laws for Neural Language Models* 第一次系统地把"loss 随 N/D/C 怎么变"刻画出来——**loss 按三轴的幂律下降,且三者各自存在临界点**。Chinchilla 2022 修正了 Kaplan 的结论:**计算预算固定时,N 和 D 应该按 1:20 同比例增长**,而 GPT-3 那种"N 大 D 小"的配置训练不足。这一修正催生了 LLaMA、Mistral 等"参数小但数据多"的开源模型。

**涌现能力**——这家族最特别的现象是**某些能力只在规模超过临界值后突然出现**:

- GPT-2(1.5B)之前几乎没有 zero-shot 能力,1.5B 之后开始能 zero-shot 翻译、摘要、QA
- GPT-3(175B)之前 in-context learning 几乎没有,175B 之后能仅靠 prompt 里的几个例子学新任务
- GPT-3.5 之后的代码生成、链式推理(CoT)等能力同样"突然涌现"

涌现现象是 LLM 时代最让人兴奋也最让人困惑的:它给"规模就是质变"提供了实证,但也意味着**靠小模型实验无法预测大模型行为**——这一不可预测性是当前 alignment / safety 的核心难点。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2018 | **GPT-1** | 第一次用 decoder-only Transformer + 无监督生成预训练 + 任务微调跑通"预训练范式",GLUE 等 8 任务上拿 SOTA | 每个 NLP 任务单独训练,没有通用语言表征学习方法 |
| 2019 | **GPT-2** | 1.5B 参数 + WebText 40B token,zero-shot 任务能力首次涌现,证明"足够大的 LM 不微调也能做任务" | GPT-1 仍需任务微调;LM 的"通用性"尚无实证 |
| 2020 | **GPT-3** | 175B 参数 + in-context learning,仅靠 prompt 里 few-shot 例子学新任务,LLM 时代正式开启 | 涌现需要继续 scale;微调仍是默认范式 |
| 2020/2022 | **Scaling Laws** | Kaplan 给出 loss 随 N/D/C 的幂律;Chinchilla 修正 N:D = 1:20 的最优比,催生数据驱动的小模型(LLaMA) | 加规模何时收益递减、最优配比是什么,只有定性认识无定量预测 |
| 2023 | **GPT-4 / LLaMA** | 多模态 + 万亿参数级闭源(GPT-4)和高质量开源(LLaMA)的双轨;现代 LLM 配方(Pre-RMSNorm + RoPE + GQA + SwiGLU)定型 | GPT-3 时代 LLM 仍主要做文本;社区缺少可复现的高质量基础模型 |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— decoder-only Transformer 的基础块
- [../05-transformer/04-rope.md](../05-transformer/04-rope.md) —— LLaMA 之后的位置编码标配
- [../05-transformer/05-flash-attention.md](../05-transformer/05-flash-attention.md) —— 现代 LLM 训练/推理的系统底座
- `../foundations/04-normalization/` —— Pre-LN / RMSNorm 在深层 Transformer 中的稳定作用

**通向哪些家族:**
- [../12-rlhf-alignment/](../12-rlhf-alignment/) —— RLHF / DPO,把基础 LLM 对齐成对话助手
- [../14-rag-agent/](../14-rag-agent/) —— 在 LLM 上做检索增强和 agent 工具调用
- [../11-peft-lora/](../11-peft-lora/) —— 参数高效微调,适配 LLM 到下游任务
- [../13-moe-efficient/](../13-moe-efficient/) —— MoE / 量化 / 蒸馏,LLM 的效率工程
- [../15-reasoning-o1-r1/](../15-reasoning-o1-r1/) —— o1 / R1 的测试时推理,LLM 能力的另一条扩展轴
