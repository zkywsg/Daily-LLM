# 反向传播与优化器模块设计

> 日期: 2026-04-02
> 状态: 待实现

## 背景

`00-Prerequisites/deep-learning-basics/README.md` 中的 2.3 节"反向传播"仅有约 10 行内容（链式法则公式 + 一句话直觉 + 一个"你要记住"），缺乏手算推导、梯度问题分析和优化器对比。需要拆为独立模块，提供完整覆盖。

## 定位

- **文件路径**: `00-Prerequisites/backpropagation/README.md`
- **类型**: 独立模块，与 `deep-learning-basics/` 并列
- **阅读顺序**: 在 `deep-learning-basics` 之后、`01-Visual-Intelligence/cnn-architectures` 之前
- **标题**: `为什么模型能从错误中学习？—— 反向传播与优化器`

## 签名元素

### 这个问题从哪来

> 1986 年，Rumelhart、Hinton、Williams 重新发现反向传播算法，让多层网络的训练成为可能。此前，人们知道链式法则，但不知道如何系统地在多层计算图上高效应用它。深度学习的"学习"，本质上就是反向传播分配误差、优化器调整参数的循环。

### 你要记住（全模块 ≤ 3 次）

1. **2.1 节后**: "反向传播不是黑科技，是链式法则的系统应用"
2. **2.4 节后**: "zero_grad → backward → step 顺序不能颠倒"
3. **工程陷阱末**: "训练崩掉先查学习率和初始化，80% 的问题在这"

### 演进笔记

> **这一技术的遗产**: 反向传播让多层网络可训练，优化器让训练收敛可控。但 MLP 对数据结构没有假设——图像的局部性和序列的时序性都被忽略。这两个盲区分别催生了卷积网络和循环网络。
>
> → 下一章: CNN 架构

## 学习目标

1. 手算一个两层网络的反向传播，写出每一步的梯度公式
2. 解释梯度消失/爆炸的成因，选择对应的初始化和裁剪策略
3. 对比 SGD、Momentum、Adam 的更新规则，知道什么时候选哪个
4. 写出完整的 `zero_grad → backward → step` 训练循环，并解释每步为什么不能少

## 内容结构（方案 A: 四段式线性递进）

### 1. 直觉（模块开头）

类比: 反向传播是"问责制"——最终结果出了错，从输出层往回逐层追问"是谁的贡献导致了这次偏差"，每层参数按责任大小接受调整。

### 2. 机制

#### 2.1 链式法则与计算图

- 标量链式法则回顾（3-4 行）
- 在两层网络上画出完整计算图（Mermaid，暖色系配色）
- 逐步手算: 从 loss 到 W₁ 的每一步偏导，写出中间梯度
- 代码: numpy 手写 `backward()`，逐步计算，最后与 `torch.autograd` 结果对比验证一致
- 依赖: numpy, torch

#### 2.2 梯度问题: 消失与爆炸

- 为什么层数加深导致梯度指数级缩小/放大（用 sigmoid 导数 ≤ 0.25 推导）
- 可视化: 不同深度网络各层梯度范数（x 轴层号，y 轴梯度 L2 范数）
- 对策（按优先级）:
  1. 激活函数选择（ReLU 替代 Sigmoid）
  2. 权重初始化（He / Xavier，解释方差稳定原理）
  3. 梯度裁剪（clip_grad_norm_，阈值选择）
- 依赖: torch, matplotlib

#### 2.3 优化器: 从 SGD 到 Adam

- SGD 更新规则 + 为什么需要学习率
- Momentum: 加入"惯性"，物理类比（球在坡上滚）
- Adam: 动量 + 自适应学习率，更新公式（m_t, v_t, bias correction）
- 代码: 同一二分类问题，SGD/Momentum/Adam 训练，画 loss 曲线对比
- 经验法则: Adam 是默认首选，SGD+Momentum 在某些 CV 任务上泛化更好
- 依赖: torch, matplotlib

#### 2.4 训练循环骨架

- `zero_grad → forward → loss → backward → step`，解释每步作用和顺序不可变原因
- lr scheduler（CosineAnnealing 或 StepLR），展示学习率变化曲线
- 代码: 完整可运行的训练循环（复用 deep-learning-basics 的 MLP 结构），含 train/eval 切换和验证集评估
- 依赖: torch

### 3. 工程陷阱（按优先级）

1. **忘记 zero_grad** → 梯度累积，等效学习率越来越大
2. **学习率设错** → 先试 1e-3，看前 10 个 batch
3. **初始化不当** → 深层网络梯度传不动
4. **Adam 的 weight_decay 陷阱** → AdamW 和 Adam+L2 不等价

## 代码规范

- 遵循 STYLE.md 渐进式实现: Step 1 核心逻辑 → Step 2 shape 安全 → Step 3 工程完善 → Step 4 生产级
- 2.1 节用 numpy 手算（不依赖 autograd），验证环节切换到 PyTorch
- 2.2-2.4 节统一用 PyTorch
- 所有随机种子 `torch.manual_seed(42)`
- 每段代码独立可运行，不依赖同文件其他代码块
- 注释头三行: 解决什么问题（不写加了什么功能）
- 格式化: Black，行宽 88

## 文件结构

```
00-Prerequisites/
├── README.md                          # 现有，目录列表中新增 backpropagation 条目
├── deep-learning-basics/
│   └── README.md                      # 现有，2.3 节末尾加链接
└── backpropagation/                   # 新建
    └── README.md                      # 主文档
```

不建独立 notebook 或 src 文件。理由:
- 当前 `00-Prerequisites/` 下无任何 .ipynb 或 .py 文件
- 反向传播的代码以推导和对比为主，README 内嵌代码块更利于阅读
- 如后续需要可从 README 提取

## 对现有文件的改动

### `00-Prerequisites/README.md`

在目录列表中增加:

```markdown
### [反向传播与优化器](backpropagation/README.md)
- 链式法则与计算图推导
- 梯度消失/爆炸与对策
- 优化器对比：SGD、Momentum、Adam
- 训练循环骨架与学习率调度
```

### `00-Prerequisites/deep-learning-basics/README.md`

在 2.3 节末尾（"你要记住"框之后）加一行:

```markdown
→ 详细推导与代码验证见 [反向传播与优化器](../backpropagation/README.md)
```

### `00-Timeline/README.md`

无需修改。1986 年反向传播早于时间线起始年份 2012，且 2012 ReLU 普及条目已覆盖梯度问题。

## 不做的事

- 不建独立 notebook（README 内嵌代码足够）
- 不建 src/ 目录（前置阶段不需要独立代码模块）
- 不覆盖自动微分内部实现细节（Jacobian-vector product 等，超出前置阶段范围）
- 不涉及二阶优化器（L-BFGS 等，与主线无关）
