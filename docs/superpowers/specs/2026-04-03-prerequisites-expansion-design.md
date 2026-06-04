# 00-Prerequisites 扩展设计

**日期**：2026-04-03
**状态**：已批准

## 背景

00-Prerequisites 现有 4 个子模块（deep-learning-basics、backpropagation、activation-functions、regularization），但下游 Phase 01-03 依赖多个未铺垫的概念。本设计填补 10 个缺口，每个缺口一个独立子模块。

## 阅读顺序

穿插在现有模块之间：

```
deep-learning-basics (已有)
  → softmax (新增 #1)
  → loss-functions (新增 #2)
backpropagation (已有)
  → normalization (新增 #3)
  → residual-connections (新增 #4)
activation-functions (已有)
regularization (已有)
  → embeddings (新增 #5)
  → tokenization (新增 #6)
  → encoder-decoder (新增 #7)
  → attention-primer (新增 #8)
  → inductive-bias (新增 #9)
  → numerical-precision (新增 #10)
```

## 模块详细规格

### #1: softmax/

**标题**：为什么多分类不能直接比大小？—— Softmax 与概率分布

**学习目标**：
1. 解释 softmax 的公式和它的"归一化成概率"直觉
2. 说明温度参数 T 如何控制分布尖锐程度
3. 手写数值稳定的 softmax（log-sum-exp trick）

**内容结构**：
- 问题从哪来：多分类需要一个"公平竞争"的概率分布
- 直觉：把任意实数变成归一化的概率
- 机制：公式推导、与 Sigmoid 的关系、温度参数、log-sum-exp trick
- 渐进式实现：纯 numpy → 稳定版 → 带温度采样
- 工程陷阱：维度搞错、input 全负仍合法但梯度行为不同
- 演进笔记：softmax 在注意力机制中的角色预告

**篇幅**：~250 行

---

### #2: loss-functions/

**标题**：怎么量化"错得有多离谱"？—— 损失函数全景

**学习目标**：
1. 区分回归、分类、对比学习三类损失的设计动机
2. 解释 Focal Loss 解决类别不平衡的原理
3. 说明 label smoothing 为什么能防止过度自信

**内容结构**：
- 问题从哪来：不同的"打分方式"教会模型不同的技能
- 直觉：损失函数是老师的评分标准
- 机制：MSE/MAE/Huber、交叉熵/BCE、Focal Loss、InfoNCE/对称交叉熵、Label Smoothing
- 渐进式实现：从 MSE 到 Focal Loss 的渐进代码
- 工程陷阱：log(0) NaN、类别不平衡的 bias
- 演进笔记：从监督损失到对比损失到 RLHF reward 的演进脉络

**篇幅**：~300 行

---

### #3: normalization/

**标题**：为什么训练深度网络需要"校准仪"？—— 归一化机制

**学习目标**：
1. 解释 BatchNorm 训练/推理的行为差异（batch stats vs running stats）
2. 说明为什么 Transformer 选 LayerNorm 而不选 BatchNorm
3. 手写 BatchNorm 前向传播（含 running mean/var 更新）

**内容结构**：
- 问题从哪来：深层网络每一层的输入分布都在变（Internal Covariate Shift）
- 直觉：归一化是流水线上的"校准仪"
- 机制：BatchNorm（含 train/eval）、LayerNorm、GroupNorm/InstanceNorm 简述、BN vs LN 对比表
- 渐进式实现：手写 BatchNorm → PyTorch 验证
- 工程陷阱：小 batch BN 统计不准、LN 的 gamma/beta 初始化
- 演进笔记：BN 催生超深 CNN → LN 催生稳定 Transformer → RMSNorm (LLaMA)

**篇幅**：~300 行

---

### #4: residual-connections/

**标题**：为什么要把输入"抄近路"送回去？—— 残差连接

**学习目标**：
1. 从梯度推导解释为什么残差连接能缓解梯度消失
2. 说明 projection shortcut 的使用场景
3. 解释 Pre-LN vs Post-LN 对训练稳定性的影响

**内容结构**：
- 问题从哪来：网络加深反而变差（退化问题，不是过拟合）
- 直觉：残差连接是"跳板"——至少无损传递原始信号
- 机制：y=x+F(x) 梯度推导、projection shortcut、DenseNet 对比
- 渐进式实现：手动残差块 → nn.Sequential 对比
- 工程陷阱：维度不匹配用 projection、Pre-LN vs Post-LN
- 演进笔记：ResNet → Transformer x+sublayer(LN(x)) → DenseNet → U-Net skip connections

**篇幅**：~250 行

---

### #5: embeddings/

**标题**：为什么"苹果"和"橘子"在向量空间里是邻居？—— Embedding 向量

**学习目标**：
1. 解释 one-hot → embedding lookup 的映射过程
2. 说明 Word2Vec 的核心思想（上下文预测 + 负采样）
3. 区分静态 embedding 和上下文 embedding

**内容结构**：
- 问题从哪来：离散符号无法做数学运算，需要连续向量表示
- 直觉：每个 token 的"语义身份证"
- 机制：one-hot → lookup、Word2Vec（Skip-gram/CBOW）、GloVe、nn.Embedding 本质
- 渐进式实现：手写简化版 Skip-gram → nn.Embedding 用法
- 工程陷阱：维度选择（128-768）、OOV 词处理
- 演进笔记：静态→上下文（ELMo→BERT）的转折点

**篇幅**：~300 行

---

### #6: tokenization/

**标题**：模型怎么"读"文字？—— 分词器

**学习目标**：
1. 解释子词级分词为什么胜出（词级和字符级的缺点）
2. 手述 BPE 算法的训练过程
3. 说明 BPE、WordPiece、Unigram 三者的核心差异

**内容结构**：
- 问题从哪来：模型只认整数，需要把文字变成 token ID
- 直觉：分词器是"翻译官"
- 机制：词级/字符级/子词级对比、BPE 算法、WordPiece、Unigram、special tokens
- 渐进式实现：手写 BPE 训练 → HuggingFace tokenizers 使用
- 工程陷阱：vocab size 权衡、中英文分词差异
- 演进笔记：从规则分词→统计分词→子词分词的演进

**篇幅**：~300 行

---

### #7: encoder-decoder/

**标题**：为什么要把"理解"和"生成"分开？—— 编码器-解码器范式

**学习目标**：
1. 解释 encoder-decoder 分工的设计动机
2. 区分 encoder-only、decoder-only、encoder-decoder 三种 Transformer 范式
3. 说明各范式适合的任务类型

**内容结构**：
- 问题从哪来：Seq2Seq 翻译的"先理解再表达"
- 直觉：翻译不是边听边说——先听完理解，再组织表达
- 机制：Seq2Seq 范式、三种 Transformer 范式及适用任务、attention mask 差异
- 渐进式实现：最简 Seq2Seq（LSTM 版）
- 工程陷阱：encoder-decoder 的 KV cache 差异
- 演进笔记：Seq2Seq → Transformer encoder-decoder → decoder-only 主流化

**篇幅**：~250 行

---

### #8: attention-primer/

**标题**：为什么模型需要"回头看"？—— 注意力机制动机

**学习目标**：
1. 解释 Seq2Seq 的瓶颈问题（固定长度向量压缩整个句子）
2. 说明 QKV 框架的直觉
3. 理解注意力的 O(n²) 复杂度来源

**内容结构**：
- 问题从哪来：Seq2Seq 的固定长度上下文向量是信息瓶颈
- 直觉：解码时"回头看"输入中最相关的部分
- 机制：QKV 框架直觉、注意力权重计算、不展开多头/不展开完整 Transformer
- 渐进式实现：最简注意力可视化
- 工程陷阱：O(n²) 复杂度问题
- 演进笔记：Bahdanau → Luong → Multi-Head Self-Attention 预告

**篇幅**：~250 行

---

### #9: inductive-bias/

**标题**：为什么 CNN 和 Transformer "看到"的世界不同？—— 归纳偏置

**学习目标**：
1. 解释归纳偏置的定义和它在模型设计中的角色
2. 对比 CNN、RNN、Transformer 各自的归纳偏置
3. 说明 ViT 如何用数据规模补偿归纳偏置的缺失

**内容结构**：
- 问题从哪来：同样的数据，不同的模型架构学到不同的东西
- 直觉：归纳偏置是模型在看到数据之前就做的假设
- 机制：参数共享、局部连接、平移不变性、CNN vs RNN vs Transformer 对比表
- 无渐进式实现（纯概念模块）
- 工程陷阱：归纳偏置越强→数据需求越少但灵活性越低
- 演进笔记：人类设计偏置 → 模型学偏置（NAS/MLP-Mixer）

**篇幅**：~200 行

---

### #10: numerical-precision/

**标题**：为什么 FP16 训练会"丢精度"？—— 数值精度与分布式训练基础

**学习目标**：
1. 解释 FP32/FP16/BF16 的位数分配和表示范围差异
2. 说明混合精度训练的工作流程（FP16 前向 + FP32 主权重 + loss scaling）
3. 描述数据并行的核心思想

**内容结构**：
- 问题从哪来：模型越来越大，训练越来越慢——精度换速度
- 直觉：FP32 是精确记账，FP16 是粗略估算但快一倍
- 机制：浮点格式对比、混合精度训练流程、GradScaler、数据并行概念
- 渐进式实现：PyTorch AMP 最小示例
- 工程陷阱：BF16 vs FP16 选择、梯度累积与 batch size 关系
- 演进笔记：单 GPU → 数据并行 → ZeRO → 流水线并行 → 张量并行

**篇幅**：~250 行

---

## 跨模块更新

### 00-Prerequisites/README.md 更新
- "本阶段内容"新增 10 个条目
- 时间线节点补充（Word2Vec 2013、BN 2015、ResNet 2015、BPE 2016 等）

### 导航链接
- 每个新模块的"上一章/下一章"指向正确的阅读顺序邻居
- 已有模块的导航链接更新（如 deep-learning-basics 的下一章从 backpropagation 改为 softmax）

### 双语
- 每个模块提供中文 README.md（主）和英文 README_EN.md（辅）
- 但此阶段可以先写中文，英文后续补齐

## 教学格式（沿用现有）

每个模块遵循：
1. 这个问题从哪来（历史动机）
2. 学习目标（3 个可回答的问题）
3. 直觉（类比 + 一句话核心）
4. 机制（公式 + mermaid 图 + 代码）
5. 渐进式实现（Step 1-4）
6. 工程陷阱（按严重度排序）
7. 演进笔记（遗产 + 留下的新问题）
8. 上一章 / 下一章导航

篇幅控制在现有模块的 60-80%（桥梁内容不需要像核心模块那么厚）。
