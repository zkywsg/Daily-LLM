# 00-Prerequisites 缺口模块设计（概率 / 线性代数 / 学习率调度）

> 日期：2026-04-04
> 状态：已批准
> 目标：为 00-Prerequisites 新增 3 个前置模块，填补下游 Phase 01-05 反复引用但未铺垫的概念缺口

## 背景

00-Prerequisites 现有 14 个模块（deep-learning-basics 到 numerical-precision），覆盖了从深度学习基础到数值精度的完整前置知识链。但下游模块中频繁出现三个未被前置覆盖的概念：

1. **概率与信息论** — 交叉熵、KL 散度、最大似然估计在 loss-functions、softmax、regularization 中反复引用，却没有前置讲解
2. **线性代数** — 矩阵乘法、转置、SVD、范数在 CNN/Transformer/Embedding 中频繁出现，但没有专门模块
3. **学习率调度与梯度优化深入** — warmup、cosine decay、gradient clipping 在 Transformer 训练中是标配，backpropagation 模块未覆盖

## 阅读顺序

三个新模块穿插在现有模块之间：

```
#0  概率与信息论（新增）
  → #1  深度学习基础（已有）
    → #2  线性代数基础（新增）
    → #3  Softmax（已有）
      → #4  损失函数（已有）
        → #5  反向传播（已有）
          → #6  学习率调度（新增）
          → #7  归一化（已有）
          → #8  残差连接（已有）
          → #9  激活函数（已有）
          → #10 正则化（已有）
            → #11 Embedding（已有）
            → #12 分词器（已有）
            → #13 编码器-解码器（已有）
            → #14 注意力机制（已有）
            → #15 归纳偏置（已有）
            → #16 数值精度（已有）
```

### 分组调整

| 分组 | 模块 |
|------|------|
| 基础概念 | #0 概率与信息论, #1 深度学习基础, #2 线性代数, #3 Softmax, #4 损失函数, #5 反向传播, #6 学习率调度 |
| 架构组件 | #7 归一化, #8 残差连接, #9 激活函数, #10 正则化 |
| NLP 桥梁 | #11 Embedding, #12 分词器, #13 编码器-解码器, #14 注意力机制 |
| 概念桥梁 | #15 归纳偏置, #16 数值精度 |

### 放置理由

- **概率与信息论 (#0)** 放最前：交叉熵、KL 散度是后续所有损失函数和 softmax 的数学基础
- **线性代数 (#2)** 放在深度学习基础之后、Softmax 之前：学线性代数之前需要先知道"什么是神经网络"，但 Softmax 和损失函数中的向量运算需要线性代数基础
- **学习率调度 (#6)** 放在反向传播之后：它是反向传播的直接延伸，先懂梯度下降再学调度策略

---

## 模块 #0：概率与信息论基础

**目录**：`00-Prerequisites/probability-information-theory/README.md`
**标题**：交叉熵从哪来？—— 概率与信息论基础
**篇幅**：~450 行

### 内容结构

#### 2.1 概率基础
- 随机变量（离散/连续）、概率分布、期望与方差
- 常见分布速览：伯努利、均匀、高斯（正态）— 各附公式 + 直觉
- 高斯分布在权重初始化和 VAE 中的角色（交叉引用）

#### 2.2 贝叶斯定理
- 条件概率 → 联合概率 → 贝叶斯公式
- 先验/似然/后验/证据，"医学检测"类比
- 深度学习体现：正则化 = 先验、MAP 估计

#### 2.3 信息熵
- "平均需要多少比特来描述一个随机变量"
- 熵公式 H(X) = -Σ p(x) log p(x)，推导直觉
- 最大熵原理：均匀分布是最"不确定"的

#### 2.4 交叉熵与 KL 散度
- 交叉熵 H(p, q) = -Σ p(x) log q(x) → 链接 loss-functions
- KL 散度 D_KL(p ‖ q) = H(p, q) - H(p) → "用 q 近似 p 的额外代价"
- 非对称性：forward vs reverse KL 直觉
- 变分推断中 KL 散度的角色（为 VAE 铺路）

#### 2.5 最大似然估计
- MLE = 找让观测数据概率最大的参数
- 数学上等价于最小化交叉熵 → 打通"概率视角"和"损失函数视角"
- MAP 估计 = MLE + 先验 → L2 正则化的概率解释

#### 2.6 互信息
- I(X; Y) = H(X) - H(X|Y) = D_KL(p(x,y) ‖ p(x)p(y))
- "知道 Y 之后 X 的不确定性减少了多少"
- 在特征选择、InfoNCE 对比损失中的应用（链接 loss-functions）

### 渐进式实现
1. 纯 NumPy 实现 — 从零写信息熵、交叉熵、KL 散度，手算小例子
2. MLE 示例 — 最大似然估计拟合高斯参数 μ 和 σ，可视化拟合过程
3. PyTorch 对接 — 展示 `F.cross_entropy` 背后的公式，证明即为刚写的实现

### 工程陷阱
1. **log(0) 爆炸** — 未做 clip 导致 NaN（最常见）
2. **KL 散度非对称性搞反** — forward/reverse 用错场景，VAE 训练崩溃
3. **浮点精度下熵计算失真** — 概率极小时 log 溢出，FP16 尤甚（链接 numerical-precision）
4. **MLE 过拟合理解偏差** — 与 L2 正则化的关系混淆

### 跨模块衔接

| 本模块概念 | 衔接模块 | 衔接点 |
|-----------|---------|-------|
| 交叉熵 | loss-functions | 交叉熵损失的数学来源 |
| KL 散度 | softmax | 温度参数与软标签的 KL 关系 |
| MLE = 最小化交叉熵 | backpropagation | 训练循环里损失函数在优化什么 |
| 贝叶斯/MAP | regularization | L2 正则化 = 高斯先验的 MAP |
| 互信息 | loss-functions | InfoNCE / 对比损失的理论基础 |
| 高斯分布 | numerical-precision | 权重初始化策略的概率依据 |

### 时间线同步

需在 `00-Prerequisites/README.md` 时间线表中增加：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 1812 | Bayes 定理（Laplace） | 逆概率推理的数学基础 |
| 1948 | Shannon 信息论 | 信息熵、互信息的数学基础 |

需在 `00-Timeline/README.md` 中补充 1948 Shannon 信息论条目。

---

## 模块 #2：线性代数基础

**目录**：`00-Prerequisites/linear-algebra/README.md`
**标题**：为什么深度学习离不开矩阵乘法？—— 线性代数基础
**篇幅**：~300 行

### 内容结构

#### 2.1 向量与矩阵基础
- 向量：方向 + 大小，向量加法的几何意义
- 矩阵乘法："行看列"的计算方法，为什么 (m,n) × (n,p) → (m,p)
- 转置：Aᵀ 的含义，(AB)ᵀ = BᵀAᵀ
- 单位矩阵、逆矩阵（概念层面，不求手动计算）

#### 2.2 矩阵乘法 = 线性变换
- 矩阵乘法的几何直觉：旋转、缩放、投影
- 神经网络每一层的本质：y = Wx + b 就是一个线性变换 + 偏置
- 全连接层 = 矩阵乘法，卷积的底层也是矩阵运算（im2col）

#### 2.3 范数
- L1 范数：曼哈顿距离，在 LASSO 正则化中的角色
- L2 范数：欧几里得距离，在 Ridge 正则化 / 权重衰减中的角色
- 无穷范数：最大绝对值，gradient clipping 的按值裁剪
- 余弦相似度：方向而非大小的度量，对比学习的基础

#### 2.4 特征值与 SVD
- 特征值/特征向量的直觉："哪些方向被放大/缩小"
- SVD = UΣVᵀ，不做数学推导，聚焦直觉：
  - "把任意矩阵拆成旋转-缩放-旋转"
  - 降维（取前 k 个奇异值）、推荐系统（协同过滤）
  - LoRA 的低秩分解思想预告

#### 2.5 广播机制
- NumPy/PyTorch 的广播规则：(3,1) + (1,4) → (3,4)
- 常见广播场景：batch 数据加偏置 (B,D) + (D,)、注意力 mask

### 渐进式实现
1. **纯 NumPy 手写矩阵乘法** — 三重循环 → 向量化 → `@` 运算符，对比速度差异
2. **线性变换可视化** — 用 2×2 矩阵变换一组点（旋转/缩放/skew），直观感受矩阵的作用
3. **SVD 降维示例** — 对一张灰度图做 SVD，只保留前 k 个奇异值重建，展示信息压缩

### 工程陷阱
1. **矩阵乘法 vs 逐元素乘法混淆**（最常见）
   现象：`A * B`（逐元素）和 `A @ B`（矩阵乘法）搞混，维度对不上或结果错误
   处置：`*` 是逐元素，`@` 或 `torch.matmul` 是矩阵乘法，永远不要混淆

2. **维度不匹配**
   现象：`(B, D)` 和 `(D, K)` 相乘搞反顺序
   处置：记住"中间维度必须相同"，`(m,n) @ (n,p)` → `(m,p)`

3. **转置忘记导致形状错误**
   现象：注意力计算中 QK^T 写成 QK，维度报错
   处置：注意力里相似度计算一定是 `Q @ K.T`，转置不可省略

4. **广播机制理解错误**
   现象：以为 `(3,) + (4,)` 能广播，实际报错
   处置：广播从右往左对齐，尾部维度必须相同或为 1

### 跨模块衔接

| 本模块概念 | 衔接模块 | 衔接点 |
|-----------|---------|-------|
| 矩阵乘法 | deep-learning-basics | 全连接层 y = Wx + b |
| 转置 | attention-primer | QK^T 相似度计算 |
| 范数 | regularization | L1/L2 正则化的数学定义 |
| 余弦相似度 | loss-functions | InfoNCE 对比损失中的相似度函数 |
| SVD / 低秩 | numerical-precision | LoRA 的低秩分解思想 |
| 广播 | softmax | batch softmax 的维度处理 |

---

## 模块 #6：学习率调度与梯度优化深入

**目录**：`00-Prerequisites/optimization-scheduling/README.md`
**标题**：学习率怎么变才能又快又稳？—— 调度与梯度控制
**篇幅**：~300 行

### 内容结构

#### 2.1 学习率为什么重要
- 直觉：学习率是"步幅"——太大在最优解附近震荡，太小收敛极慢
- 可视化：不同学习率下的 loss 曲线（太高→发散，太低→平线，刚好→快速收敛）
- 固定学习率的根本问题：初期需要大步探索，后期需要小步精细调优

#### 2.2 学习率调度策略
- **Step Decay**：每隔 N 个 epoch 乘以 γ（如 ×0.1），简单粗暴
- **Cosine Annealing**：学习率沿余弦曲线从初始值降到最低值，平滑且效果好
  - 公式：η_t = η_min + 0.5(η_max - η_min)(1 + cos(πt/T))
  - 为什么 cosine 比 step decay 更受欢迎：没有"台阶"导致的梯度突变
- **Warmup**：训练初期线性/指数从小学习率升到目标值
  - 为什么 Transformer 需要 warmup：Adam 的自适应统计量在初期不稳定，大学习率会炸
  - 常见方案：线性 warmup + cosine decay
- **One-Cycle Policy**：先升后降，单周期完成训练
- **ReduceOnPlateau**：验证 loss 停滞时自动降低学习率

#### 2.3 梯度裁剪
- **Gradient Clipping by Norm**：梯度向量的 L2 范数超过阈值时等比缩放
  - 公式：if ‖g‖ > max_norm: g = g × max_norm / ‖g‖
  - RNN/LSTM 训练的标准操作（BPTT 梯度爆炸对策）
- **Gradient Clipping by Value**：每个梯度分量截断到 [-v, v]
  - 更简单但不如按范数裁剪——改变了梯度方向
- 两种方式的选择：NLP 任务按范数裁剪（max_norm=1.0），CV 任务通常不需要

#### 2.4 优化器进阶
- **AdamW**：权重衰减从梯度更新中解耦
  - Adam 的 L2 正则化实际效果 ≠ 权重衰减（因为自适应学习率缩放）
  - AdamW 把 weight decay 直接乘在权重上，不经过梯度
  - Transformer 训练的事实标准
- **LAMB**（Layer-wise Adaptive Moments）：大 batch 训练的优化器
  - 解决问题：batch size 从 256→8192 时 Adam 会发散
- **Lookahead**（"Lazy Adam"）：慢权重 + 快权重，减少优化器的方差
  - 不深入实现，只讲动机和直觉

#### 2.5 训练配置速查
- Transformer 标准配置：AdamW + 线性 warmup (1-5% steps) + cosine decay
- CNN 标准配置：SGD + Momentum + Step Decay
- 微调场景：较小学习率 + 短 warmup + 少量 epoch

### 渐进式实现
1. **手写 Cosine Decay** — NumPy 实现学习率调度器，画出不同参数下的调度曲线
2. **PyTorch `lr_scheduler` 对比** — `CosineAnnealingLR`、`OneCycleLR`、`LinearWarmup` 验证
3. **Gradient Clipping 实验** — 对比有无梯度裁剪的 RNN 训练稳定性，可视化梯度范数
4. **AdamW vs Adam** — 在 Transformer 微调场景下对比两者 loss 曲线差异

### 工程陷阱
1. **warmup 步数设错导致训练崩**（最常见）
   现象：Transformer 训练前几个 step loss 直接 NaN 或飞升
   处置：warmup steps 通常设为总步数的 1-5%，或至少 500-1000 步

2. **scheduler.step() 调用位置错误**
   现象：在 `optimizer.step()` 之前调用 `scheduler.step()`，学习率更新时机错位
   处置：PyTorch 标准顺序是 `loss.backward() → optimizer.step() → scheduler.step()`

3. **gradient accumulation 与 lr schedule 的配合**
   现象：gradient accumulation 模拟大 batch，但 lr schedule 按实际 step 更新，学习率下降太快
   处置：scheduler step 应该在 accumulation 完成后调用，或按 effective step 计算

4. **微调时学习率太大**
   现象：预训练模型微调几个 epoch 就灾难性遗忘
   处置：微调学习率通常是预训练的 1/10 到 1/100（如 5e-5 vs 3e-4）

### 跨模块衔接

| 本模块概念 | 衔接模块 | 衔接点 |
|-----------|---------|-------|
| Warmup | Transformer 架构 | 原始论文使用 warmup 的原因 |
| AdamW | backpropagation | Adam 的续篇，权重衰减的正确方式 |
| Gradient Clipping | RNN/循环网络 | BPTT 梯度爆炸的标准对策 |
| Cosine Decay | numerical-precision | 混合精度训练中的调度配合 |
| Learning Rate | 03-Scale-Multimodal | Scaling Laws 中计算量与 lr 的关系 |

### 时间线同步

需在 `00-Prerequisites/README.md` 时间线表中增加：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2017 | AdamW (Loshchilov & Hutter) | 权重衰减解耦，Transformer 训练事实标准 |
| 2017 | One-Cycle Policy (Smith) | 单周期学习率调度，训练效率提升 |
| 2019 | LAMB (You et al.) | 大 batch 训练优化器，支持 batch size 8K+ |

---

## 00-Prerequisites/README.md 更新

### 本阶段内容表更新

"基础概念"分组从 4 个扩展到 7 个，所有模块重新编号：

```
基础概念:
  #0  概率与信息论（新增）
  #1  深度学习基础（已有，原 #1）
  #2  线性代数基础（新增）
  #3  Softmax（已有，原 #2）
  #4  损失函数（已有，原 #3）
  #5  反向传播（已有，原 #4）
  #6  学习率调度（新增）

架构组件:
  #7  归一化（已有，原 #5）
  #8  残差连接（已有，原 #6）
  #9  激活函数（已有，原 #7）
  #10 正则化（已有，原 #8）

NLP 桥梁:
  #11 Embedding（已有，原 #9）
  #12 分词器（已有，原 #10）
  #13 编码器-解码器（已有，原 #11）
  #14 注意力机制（已有，原 #12）

概念桥梁:
  #15 归纳偏置（已有，原 #13）
  #16 数值精度（已有，原 #14）
```

### 建议阅读顺序更新

```
基础概念: 0 → 1 → 2 → 3 → 4 → 5 → 6
架构组件: 7 → 8 → 9 → 10
NLP 桥梁: 11 → 12 → 13 → 14
概念桥梁: 15 → 16
```

### 时间线节点更新

新增 4 个条目：1812 Bayes、1948 Shannon、2017 AdamW、2017 One-Cycle、2019 LAMB

### 导航链接更新

每个新模块的"上一章/下一章"指向正确的阅读顺序邻居。已有模块的导航链接需要更新（如 deep-learning-basics 的下一章从 softmax 改为线性代数）。

---

## 跨模块更新汇总

| 目标文件 | 变更 |
|---------|------|
| `00-Prerequisites/probability-information-theory/README.md` | 新建，~450 行 |
| `00-Prerequisites/linear-algebra/README.md` | 新建，~300 行 |
| `00-Prerequisites/optimization-scheduling/README.md` | 新建，~300 行 |
| `00-Prerequisites/README.md` | 更新分组、编号、阅读顺序、时间线 |
| `00-Timeline/README.md` | 补充 1948 Shannon 信息论条目 |
| 已有模块导航链接 | 更新"上一章/下一章"指向 |

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
