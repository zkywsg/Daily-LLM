# 设计文档：ReLU 与激活函数家族模块

**日期**：2026-04-02
**状态**：已批准
**模块路径**：`00-Prerequisites/activation-functions/`

## 1. 定位

新建独立模块 `00-Prerequisites/activation-functions/`，位于 `deep-learning-basics` 之后、`01-Visual-Intelligence` 之前。

**设计决策**：选择方案 A（ReLU 叙事主线 + 变体作演进分支）。理由：前置模块的目标是打基础，不是做激活函数百科全书。ReLU 是主角，变体知道存在和适用场景即可。

## 2. 文件清单

| 文件 | 语言 | 说明 |
|------|------|------|
| `00-Prerequisites/activation-functions/README.md` | 中文 | 主文件 |
| `00-Prerequisites/activation-functions/README_EN.md` | 英文 | 翻译版 |

## 3. 章节结构

**标题**：`# 为什么 Sigmoid 不够用了？—— ReLU 与激活函数家族`

### 3.1 签名三元素

1. **这个问题从哪来**（模块开头）
   - 2010 年 Nair & Hinton 首次提出 ReLU
   - 2012 年 AlexNet 用 ReLU 替代 Sigmoid，训练速度提升 6 倍
   - 引出：Sigmoid 到底哪里不行？

2. **你要记住**（最多 3 处，插在关键知识点后）
   - ReLU 的核心优势不是"简单"，而是正区间梯度恒为 1
   - Dying ReLU 是 ReLU 唯一的硬伤，所有变体都是为解决它而生
   - GELU 是现代 Transformer 的默认选择，但原理不同于 ReLU 家族

3. **演进笔记**（模块结尾）
   - 技术遗产：从 AlexNet 到 GPT，激活函数创新从未停止
   - 链接到 `deep-learning-basics`（上一章）
   - 链接到 `01-Visual-Intelligence/cnn-architectures/`（下一章）
   - 链接到 `02-Language-Transformers/transformer-architecture/`（GELU 详解）

### 3.2 学习目标

1. 为什么 ReLU 的梯度比 Sigmoid 更适合深度网络？
2. Dying ReLU 是什么？怎么解决？
3. 面对一个新任务，该选哪个激活函数？

### 3.3 正文小节

```
## 1. 直觉
  - 类比：Sigmoid 像一个会"疲倦"的工人（越深入越没力气）
  - ReLU 像一扇单向门——正数直接通过，负数关门

## 2. 机制
  ### 2.1 Sigmoid 的梯度消失问题
    - 公式：σ(x) = 1/(1+e^(-x))
    - 梯度：σ'(x) = σ(x)(1-σ(x))，最大值仅 0.25
    - Mermaid 图：多层链式乘法的梯度指数衰减

  ### 2.2 ReLU：定义、梯度、为什么有效
    - f(x) = max(0, x)
    - 梯度：正区间恒为 1，负区间恒为 0
    - 稀疏激活：约 50% 神经元输出为零，天然正则化
    - Mermaid 计算流图（暖色调：输入 #fef3c7/#d97706，计算 #fce7f3/#db2777，输出 #ecfdf5/#059669）

  ### 2.3 Dying ReLU 问题
    - 条件：大梯度更新将权重推入负区间 → 梯度永久为零 → 神经元"死亡"
    - 高学习率 + 无边界是主要诱因
    - 与权重初始化的耦合：ReLU 网络必须用 He 初始化

  ### 2.4 激活函数家族速览（演进分支）
    每个变体格式：1 段动机 → 1 行公式 → 1 行 PyTorch 代码

    | 变体 | 动机 | 公式 | 代码 |
    |------|------|------|------|
    | Leaky ReLU | 给负区间一个小斜率，避免 Dying ReLU | f(x)=x if x>0, αx otherwise | nn.LeakyReLU(0.01) |
    | PReLU | 斜率可学习，让网络自己决定 | 同上，α 是参数 | nn.PReLU() |
    | ELU | 负区间平滑趋近 -α，输出均值接近零 | f(x)=x if x>0, α(e^x-1) otherwise | nn.ELU() |
    | SELU | 自归一化，配合特定初始化使用 | 缩放版 ELU | nn.SELU() |
    | GELU | 按激活值概率门控，平滑版 ReLU | x·Φ(x)，Φ 是标准正态 CDF | nn.GELU() |

## 3. 渐进式实现（4 步）

  ### Step 1：ReLU 核心逻辑（5-10 行）
    - 纯 Python 实现 max(0, x)
    - 验证梯度行为

  ### Step 2：+ Sigmoid vs ReLU 梯度对比可视化
    - 画 10 层网络中梯度随层数的衰减曲线
    - matplotlib：Sigmoid 指数衰减 vs ReLU 恒定

  ### Step 3：+ Dying ReLU 复现实验
    - 模拟：大学习率更新后神经元永久输出零
    - 统计"死亡率"

  ### Step 4：+ 全家族性能 benchmark
    - 同一网络，不同激活函数，对比训练速度和最终精度
    - 输出对比表格

## 4. 工程陷阱（按严重度排序）

  1. 学习率太大 → Dying ReLU（最常见，最致命）
  2. ReLU 网络用 Xavier 初始化 → 应该用 He 初始化
  3. 输出层用 ReLU → 输出无界，用 Sigmoid/Softmax
  4. GELU 推理比 ReLU 慢 → 生产环境考虑近似或缓存

## 演进笔记
  - ReLU 的遗产：从 AlexNet 到 GPT 都在用激活函数创新
  - 链接：上一章 → deep-learning-basics
  - 链接：下一章 → 01-Visual-Intelligence/cnn-architectures/
  - 链接：GELU 详见 → 02-Language-Transformers/transformer-architecture/

## 页脚
  **上一章**: [深度学习基础](../deep-learning-basics/README.md) | **下一章**: [CNN 架构](../../01-Visual-Intelligence/cnn-architectures/README.md)
```

## 4. 时间线与导航同步

### 4.1 时间线同步

**文件**：`00-Timeline/README.md`

- 2012 年 "ReLU 普及" 条目已存在
- 操作：在该条目后添加模块链接 `[→ 详解](../00-Prerequisites/activation-functions/README.md)`

### 4.2 导航更新

| 文件 | 操作 |
|------|------|
| `00-Prerequisites/README.md` | 在模块列表中添加 activation-functions，位于 deep-learning-basics 之后 |
| `00-Prerequisites/deep-learning-basics/README.md` | 演进笔记中添加指向 activation-functions 的链接 |
| `00-Prerequisites/deep-learning-basics/README_EN.md` | 同上（英文版） |
| `00-Prerequisites/activation-functions/README.md` | 页脚：上一章 → deep-learning-basics，下一章 → cnn-architectures |

### 4.3 不做的事

- 不修改下游模块（CNN、Transformer 等）中现有的 `nn.ReLU()` 使用
- 不修改 `STYLE.md`
- 不修改其他模块的演进笔记链接

## 5. 风格规范

严格遵循 `STYLE.md`：

- 标题格式：`# 为什么 [旧方法] 不够用了？—— [技术名]`
- Mermaid 暖色调（输入 #fef3c7/#d97706，计算 #fce7f3/#db2777，输出 #ecfdf5/#059669，问题 #fff7ed/#ea580c，演进 #eff6ff/#2563eb）
- 代码：3 行注释头，`torch.manual_seed(42)`，docstring 带 shape，Black 88 字符
- 渐进式实现 4 步：核心 → 边界 → 工程 → 生产
- 签名元素：`这个问题从哪来`（开头）、`你要记住`（最多 3 处）、`演进笔记`（结尾）
