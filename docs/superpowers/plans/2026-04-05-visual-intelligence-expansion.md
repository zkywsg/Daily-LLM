# 01-Visual-Intelligence 扩展与优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 01-Visual-Intelligence 新增 3 个主题模块（object-detection, segmentation-gan, lightweight-vision），修复现有结构问题（导航顺序、英文翻译、迁移提示），并为所有模块添加 PyTorch 代码练习。

**Architecture:** 3 个新子目录各含 README.md（中文）+ README_EN.md（英文）+ exercises/。修复 4 个现有文件（章节 README.md、training/README_EN.md、sequence-models/ 两个 README）。更新所有模块的底部导航链接，统一为 Training→CNN→Detection→Seg/GAN→Lightweight 顺序。

**Tech Stack:** Markdown + LaTeX (KaTeX) + Mermaid 图表 + Python 3 (PyTorch ≥ 2.0)

**Design Spec:** `docs/superpowers/specs/2026-04-05-visual-intelligence-expansion-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `01-Visual-Intelligence/README.md` | 修改 | 统一导航顺序：Training→CNN→Detection→Seg/GAN→Lightweight→Sequence |
| `01-Visual-Intelligence/README_EN.md` | 新建 | 章节级英文概览，翻译 README.md |
| `01-Visual-Intelligence/training/README_EN.md` | 重写 | 匹配中文版叙事式教学格式 |
| `01-Visual-Intelligence/sequence-models/README.md` | 修改 | 更新迁移提示措辞 |
| `01-Visual-Intelligence/sequence-models/README_EN.md` | 重写 | 匹配中文迁移提示，修复 03-NLP-Transformers 断链 |
| `01-Visual-Intelligence/object-detection/README.md` | 新建 | 目标检测模块（中文） |
| `01-Visual-Intelligence/object-detection/README_EN.md` | 新建 | 目标检测模块（英文） |
| `01-Visual-Intelligence/object-detection/exercises/anchor_boxes.py` | 新建 | Anchor 生成与 IoU 计算 |
| `01-Visual-Intelligence/object-detection/exercises/yolo_loss.py` | 新建 | YOLO 损失函数 |
| `01-Visual-Intelligence/segmentation-gan/README.md` | 新建 | 分割与生成模块（中文） |
| `01-Visual-Intelligence/segmentation-gan/README_EN.md` | 新建 | 分割与生成模块（英文） |
| `01-Visual-Intelligence/segmentation-gan/exercises/unet.py` | 新建 | U-Net 实现 |
| `01-Visual-Intelligence/segmentation-gan/exercises/dcgan_generator.py` | 新建 | DCGAN 生成器 |
| `01-Visual-Intelligence/lightweight-vision/README.md` | 新建 | 轻量化架构模块（中文） |
| `01-Visual-Intelligence/lightweight-vision/README_EN.md` | 新建 | 轻量化架构模块（英文） |
| `01-Visual-Intelligence/lightweight-vision/exercises/depthwise_separable.py` | 新建 | 深度可分离卷积对比 |
| `01-Visual-Intelligence/training/exercises/training_loop.py` | 新建 | 训练循环练习 |
| `01-Visual-Intelligence/cnn-architectures/exercises/conv_basics.py` | 新建 | 卷积基础练习 |
| `01-Visual-Intelligence/cnn-architectures/exercises/residual_block.py` | 新建 | 残差块练习 |

### 阅读顺序（最终）

```
Training → CNN Architectures → Object Detection → Segmentation & GAN → Lightweight Vision → (02-Language-Transformers)
                                                                  ↘ sequence-models (迁移提示页)
```

### 模块间导航链接模板

每个模块 README 底部的导航链接格式：

```markdown
---

**上一章**：[模块N-1名称](../module-n-1/README.md) | **下一章**：[模块N+1名称](../module-n+1/README.md)
```

具体映射：

| 模块 | 上一章 | 下一章 |
|------|--------|--------|
| training | [深度学习基础](../../00-Prerequisites/deep-learning-basics/README.md) | [CNN 架构](../cnn-architectures/README.md) |
| cnn-architectures | [训练与优化](../training/README.md) | [目标检测](../object-detection/README.md) |
| object-detection | [CNN 架构](../cnn-architectures/README.md) | [分割与生成](../segmentation-gan/README.md) |
| segmentation-gan | [目标检测](../object-detection/README.md) | [轻量化架构](../lightweight-vision/README.md) |
| lightweight-vision | [分割与生成](../segmentation-gan/README.md) | [语言线](../../02-Language-Transformers/README.md) |

---

## 教学格式模板（所有新模块统一遵循）

每个中文 README 必须包含以下结构（参考 `training/README.md` 和 `cnn-architectures/README.md` 的写法）：

```
# [问题驱动式标题]

## 这个问题从哪来
> [1-2 段历史背景，交代时间、人物、关键论文]

## 学习目标
完成本章后，你应能回答：
1. ...
2. ...
3. ...

---

## 1. 直觉
[不用公式的直觉解释，可用生活比喻]

> 你要记住：[一句话核心要点]

---

## 2. 机制
### 2.1 [子主题 1]
[公式 + 解释]
### 2.2 [子主题 2]
...
### 2.N 渐进式实现
[Step 1 · 最简版 → Step 2 · 加入关键组件 → ... → Step N · 生产级]

---

## 3. 工程要点
1. **[要点 1]** → 处置：[做法]
2. ...

> 你要记住：[排查优先级总结]

---

## 演进笔记
> **这一技术的遗产**：[承前启后的 2-3 段论述]

→ 下一章：[链接]

---

**上一章**：[链接] | **下一章**：[链接]
```

英文版结构完全对应，翻译所有内容。

---

### Task 1: 修复章节导航顺序

**Files:**
- Modify: `01-Visual-Intelligence/README.md`

当前章节 README 中模块列表顺序为 CNN→Sequence→Training，需改为 Training→CNN→Object Detection→Seg/GAN→Lightweight→Sequence。

- [ ] **Step 1: 重排 README.md 中的模块列表**

把现有 `## 本阶段内容` 下的三个模块条目替换为以下内容（保留时间线节点和底部导航不变）：

```markdown
## 本阶段内容

### [训练与优化](training/README.md)
Dropout、Batch Norm、数据增强、GPU 训练技巧
- 正则化：Dropout、DropConnect（原理详见 [前置·正则化](../00-Prerequisites/regularization/README.md)）
- 归一化：Batch Norm、Layer Norm
- 优化器：SGD、Adam
- 训练稳定性工程

### [CNN 架构](cnn-architectures/README.md)
从"图像为什么不能直接交给全连接层"出发，沿着 AlexNet → ResNet 的问题链理解经典 CNN 演进，并收束到注意力出现前的局部建模边界。
- 卷积、感受野与下采样
- 深度、计算量与信息流动的权衡
- 经典 CNN 如何一步步逼近注意力时代

### [目标检测](object-detection/README.md)
从"分类一张图"到"找到图里所有东西"——检测范式从两阶段到单阶段的演进。
- R-CNN → Faster R-CNN 的区域提案革命
- YOLO → SSD 的单阶段检测与多尺度策略
- RetinaNet / Focal Loss 解决类别不平衡

### [分割与生成](segmentation-gan/README.md)
理解像素 vs 创造像素——编码器-解码器架构的分割与生成两副面孔。
- FCN / U-Net 的语义分割
- GAN / DCGAN / Progressive GAN 的生成对抗
- Neural Style Transfer 的内容与风格分离

### [轻量化架构](lightweight-vision/README.md)
大模型很好，但手机装不下——2016-2017 年的模型效率革命。
- SqueezeNet / MobileNet 的参数压缩策略
- SE-Net 的通道注意力
- 深度可分离卷积与 CNN 路线收尾

### [序列模型](sequence-models/README.md)
RNN / LSTM / GRU / Seq2Seq 主章节已迁移到 [语言线·循环神经网络与 Seq2Seq](../02-Language-Transformers/recurrent-networks/README.md)
- 旧页暂保留用于历史访问与参考
- 该内容仍可作为视觉线下的过渡入口
```

- [ ] **Step 2: 验证 Markdown 渲染**

确认链接路径正确（所有相对路径指向已存在或即将创建的文件）。

- [ ] **Step 3: Commit**

```bash
git add 01-Visual-Intelligence/README.md
git commit -m "nav: reorder 01-Visual-Intelligence module list to reading order"
```

---

### Task 2: 创建章节级 README_EN.md

**Files:**
- Create: `01-Visual-Intelligence/README_EN.md`

- [ ] **Step 1: 创建英文版章节概览**

翻译 `01-Visual-Intelligence/README.md` 的全部内容为英文。结构完全对应中文版：

- 标题: `# Phase 01 · Vision Line (2012–2017)`
- 导言段落
- 模块列表（6 个，顺序与中文版一致，链接指向各模块的 `README_EN.md`）
- 时间线节点表格
- 底部导航到 02-Language-Transformers

顶部加语言切换：`**[English](README_EN.md) | [中文](README.md)**`

每个模块链接指向英文版：如 `[Training & Optimization](training/README_EN.md)`

- [ ] **Step 2: Commit**

```bash
git add 01-Visual-Intelligence/README_EN.md
git commit -m "content: add English overview for 01-Visual-Intelligence chapter"
```

---

### Task 3: 重写 training/README_EN.md

**Files:**
- Rewrite: `01-Visual-Intelligence/training/README_EN.md`

当前英文版是旧的百科式参考格式（~500 行），需重写为与中文版匹配的叙事式教学格式（参考 `cnn-architectures/README_EN.md` 的翻译质量）。

- [ ] **Step 1: 重写为叙事式格式**

完全翻译 `training/README.md` 的中文内容，包含：

1. 标题: `# Why Does Making Networks Deeper Make Training Worse? — Optimization & Training Stability`
2. `## Where This Problem Came From` — 翻译 AlexNet/Krizhevsky → BatchNorm/Ioffe-Szegedy 的历史叙述
3. `## Learning Goals` — 3 个学习目标
4. `## 1. Intuition` — 骑自行车比喻 + BatchNorm/Dropout 直觉
5. `## 2. Mechanism` — 4 个子节 + 4 个渐进式 PyTorch 代码示例（Step 1-4）
6. `## 3. Engineering Pitfalls` — 5 个要点
7. `## Evolution Notes` — 过渡到 CNN
8. 底部导航链接

保持 Mermaid 图表、LaTeX 公式、代码块完整翻译。代码中的注释也翻译为英文。

- [ ] **Step 2: 验证格式一致性**

对照中文版检查：所有小节标题、公式、代码块、Mermaid 图表均有对应。

- [ ] **Step 3: Commit**

```bash
git add 01-Visual-Intelligence/training/README_EN.md
git commit -m "content: rewrite training README_EN to match narrative format"
```

---

### Task 4: 修复 sequence-models 迁移提示

**Files:**
- Modify: `01-Visual-Intelligence/sequence-models/README.md`
- Rewrite: `01-Visual-Intelligence/sequence-models/README_EN.md`

- [ ] **Step 1: 更新中文版迁移提示**

当前 `sequence-models/README.md` 的迁移提示已经是中文且指向正确路径，但底部导航链接需要更新：

将底部：
```
**上一篇**：[CNN 架构](../cnn-architectures/README.md) | **下一篇**：[训练](../training/README.md)
```
改为：
```
**返回**：[Phase 01 概览](../README.md) | **主章节**：[循环神经网络与 Seq2Seq](../../02-Language-Transformers/recurrent-networks/README.md)
```

- [ ] **Step 2: 重写英文版为迁移提示**

将 `sequence-models/README_EN.md` 从旧百科式内容（~217 行）替换为与中文版匹配的迁移提示页：

```markdown
# Sequence Models

**[English](README_EN.md) | [中文](README.md)**

> Note: The RNN / LSTM / GRU / Seq2Seq main chapter has been migrated to [Language Line · Recurrent Networks & Seq2Seq](../../02-Language-Transformers/recurrent-networks/README.md). This page is retained for historical reference and as a transitional entry point from the Vision Line.

The content that was previously here covered:
- Basic RNN mechanics and vanishing/exploding gradients
- LSTM gates and cell state
- GRU simplified gating
- Sequence-to-sequence architecture
- Transition to attention mechanisms

All of this material is now available in expanded form at the link above.

---

**Back**: [Phase 01 Overview](../README_EN.md) | **Main Chapter**: [Recurrent Networks & Seq2Seq](../../02-Language-Transformers/recurrent-networks/README.md)
```

- [ ] **Step 3: 验证断链修复**

确认英文版不再包含 `../../03-NLP-Transformers/attention-mechanisms/README.md` 这个错误路径。

- [ ] **Step 4: Commit**

```bash
git add 01-Visual-Intelligence/sequence-models/README.md 01-Visual-Intelligence/sequence-models/README_EN.md
git commit -m "fix: update sequence-models migration notices and fix broken links"
```

---

### Task 5: 创建 object-detection/README.md（中文）

**Files:**
- Create: `01-Visual-Intelligence/object-detection/README.md`

- [ ] **Step 1: 编写完整中文 README（~500-600 行）**

按照教学格式模板，内容结构如下：

```
# 图像里到底有什么？—— 目标检测的演进（2014–2017）

## 这个问题从哪来
> 2014 年之前，CNN 已经能准确分类 ImageNet 图片，但分类只回答"这张图是什么"。
> 现实世界需要更多：自动驾驶要知道行人在哪里，医学影像要定位病灶有几个、有多大。
> Girshick et al. (2014) 提出 R-CNN，开启了"在图像中找物体"的检测范式。
> 此后三年，检测框架从两阶段到单阶段、从慢到快、从粗到精，演进了完整的一代。

## 学习目标
1. 两阶段检测（R-CNN → Fast → Faster）解决了什么问题，每代改进了什么瓶颈？
2. YOLO 为什么能把检测变成回归问题，单阶段检测的 trade-off 是什么？
3. Focal Loss 解决的类别不平衡问题的数学本质是什么？

## 1. 直觉
想象你在人群中找一个朋友。
方法一：先看每个人的脸（区域提案），再判断是不是他（分类）。这是两阶段。
方法二：一眼扫过去，同时定位和识别。这是单阶段。
方法一更准但慢，方法二快但容易漏。目标检测的核心张力就在这里。

> 你要记住：检测 = 定位 + 分类。两阶段拆开做，单阶段合在一起。

## 2. 机制

### 2.1 两阶段检测：从滑窗到区域提案
- R-CNN (2014): Selective Search 提候选 → CNN 提特征 → SVM 分类。慢（~47s/图）。
- Fast R-CNN (2015): 整图过 CNN → ROI Pooling → 全连接分类。加速到 0.3s。
- Faster R-CNN (2015): RPN 替代 Selective Search，端到端。关键洞察——"候选框本身也可以用网络学"。

RPN 核心公式：
- 在每个位置生成 k 个 anchor，预测 (dx, dy, dw, dh) 偏移量
- 回归目标：t_x = (x - x_a) / w_a, t_y = (y - y_a) / h_a 等

用 Mermaid 图展示 R-CNN → Fast → Faster 的演进对比。

### 2.2 单阶段检测：YOLO
YOLO v1 (Redmon et al., 2016) 核心思想：
- 把图分成 S×S 网格
- 每个网格预测 B 个框 + C 个类别概率
- 输出张量：S × S × (B × 5 + C)
- 一个 forward pass 完成检测

Loss 函数（5 项之和）：
- 坐标损失（宽高用平方根）
- 置信度损失（有物体/无物体）
- 分类损失
- 无物体置信度损失权重降低（λ_noobj = 0.5）

### 2.3 多尺度检测：SSD
SSD (Liu et al., 2016) 解决的问题：YOLO 用最后一层特征图做检测，小物体检测差。
- 在多个尺度的特征图上做检测
- 低分辨率特征图检测大物体，高分辨率检测小物体
- Default box 类似 anchor 但跨越多层

### 2.4 类别不平衡：Focal Loss / RetinaNet
核心问题：背景框远多于前景框（典型比例 1000:1），简单负样本淹没梯度。

Focal Loss 公式：
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- γ > 0 降低 easy example 的 loss 权重
- γ = 0 退化为标准交叉熵
- 实践中 γ = 2 效果最好

### 2.5 渐进式实现

Step 1 · IoU 计算（检测的基础度量）:
```python
import torch

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """boxes: (N, 4) xyxy format"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter)
```

Step 2 · Anchor 生成:
```python
def generate_anchors(feature_size, stride, scales, ratios):
    """生成基础 anchor 网格"""
    anchors = []
    for y in range(feature_size):
        for x in range(feature_size):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for s in scales:
                for r in ratios:
                    h = s * torch.sqrt(torch.tensor(r))
                    w = s / torch.sqrt(torch.tensor(r))
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return torch.tensor(anchors)
```

Step 3 · NMS (非极大值抑制):
```python
def nms(boxes, scores, threshold=0.5):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])[0]
        mask = ious <= threshold
        order = order[1:][mask]
    return keep
```

## 3. 工程要点
1. **Anchor 设计不当** → 覆盖不住目标尺寸 → Recall 低
   处置：在训练集上 K-means 聚类真实框尺寸，生成数据驱动的 anchor
2. **NMS 阈值太严** → 密集物体被误抑制（如一群人）
   处置：Soft-NMS 或调高阈值；YOLO 系列 0.3-0.5 之间调
3. **小物体漏检** → 特征图分辨率不够
   处置：用高分辨率特征图层（FPN 的 P3/P4）
4. **mAP 评估** → IoU 阈值选择影响巨大
   处置：COCO 标准 mAP@[0.5:0.95]，不只是 mAP@0.5
5. **训练不收敛** → 检测头梯度不稳定
   处置：先冻结 backbone 训练检测头 2-3 epoch，再解冻全模型微调

> 你要记住：检测 pipeline 的排障顺序是 Anchor 设计 → NMS → 特征图尺度 → Loss 权重 → Backbone 微调策略。

## 演进笔记
> 检测框架从手工设计 anchor 到 anchor-free（CornerNet, CenterNet），
> 从单尺度到多尺度（FPN），再到 DETR 用 Transformer 端到端做检测。
> 检测的演进主线是"减少手工设计、增加端到端学习"。
> Focal Loss 的思想后来也被借鉴到其他极端不平衡场景。

→ 下一章：[分割与生成 — 编码器-解码器的两副面孔](../segmentation-gan/README.md)

---

**上一章**：[CNN 架构](../cnn-architectures/README.md) | **下一章**：[分割与生成](../segmentation-gan/README.md)
```

- [ ] **Step 2: 验证格式**

检查：所有 LaTeX 公式、代码块、Mermaid 图表格式正确。

- [ ] **Step 3: Commit**

```bash
git add 01-Visual-Intelligence/object-detection/README.md
git commit -m "content: add object-detection module (Chinese)"
```

---

### Task 6: 创建 object-detection/README_EN.md（英文）

**Files:**
- Create: `01-Visual-Intelligence/object-detection/README_EN.md`

- [ ] **Step 1: 翻译中文 README 为英文**

完全翻译 Task 5 创建的中文版。要求：
- 顶部语言切换：`**[English](README_EN.md) | [中文](README.md)**`
- 标题: `# What's Actually in the Image? — Object Detection Evolution (2014–2017)`
- 所有小节标题、正文、公式注释、代码注释翻译为英文
- LaTeX 公式和代码逻辑保持不变
- 导航链接指向英文版（`../cnn-architectures/README_EN.md` 等）

- [ ] **Step 2: Commit**

```bash
git add 01-Visual-Intelligence/object-detection/README_EN.md
git commit -m "content: add object-detection module (English)"
```

---

### Task 7: 创建 segmentation-gan/README.md（中文）

**Files:**
- Create: `01-Visual-Intelligence/segmentation-gan/README.md`

- [ ] **Step 1: 编写完整中文 README（~550-650 行）**

按照教学格式模板，内容结构如下：

```
# 理解像素还是创造像素？—— 分割与生成（2015–2017）

## 这个问题从哪来
> CNN 已经能分类和检测了，但分类输出一个标签，检测输出几个矩形框。
> 现实需要更精细的回答：自动驾驶要精确到每个像素属于道路还是人行道（语义分割）。
> 与此同时，Goodfellow et al. (2014) 提出了 GAN：如果判别器和生成器对抗训练，
> 网络能否学会"创造"逼真的图像？
> 分割和生成看似无关，但它们共享同一种架构——编码器-解码器。

## 学习目标
1. FCN 和 U-Net 如何把分类网络改造为逐像素预测？
2. GAN 的博弈训练框架为什么能工作，训练不稳定的原因是什么？
3. 编码器-解码器架构如何在分割和生成中分别发挥作用？

## 1. 直觉
想象一个画家和鉴定师。
鉴定师看过很多真画，能分辨真假。画家不断画，试图骗过鉴定师。
两人一起进步，画家的画越来越像真的。
GAN 的 Generator 就是画家，Discriminator 就是鉴定师。

分割的直觉更像填色游戏：给你一张图，不是涂一种颜色，而是给每个像素选正确的颜色（类别）。

> 你要记住：分割是"每个像素都分类"，生成是"从噪声中采样出真实分布"。两者共享编码器-解码器结构。

## 2. 机制

### 2.1 语义分割：FCN
Long et al. (2015) 的核心洞察：把分类网络的全连接层换成卷积层。
- 标准 CNN: 卷积 → 池化 → 全连接 → 类别概率
- FCN: 卷积 → 池化 → 1×1 卷积 → 上采样 → 逐像素类别概率

关键操作：转置卷积（反卷积）用于上采样
$$Y(i,j) = \sum_m \sum_n X(\lfloor i/s \rfloor + m,\, \lfloor j/s \rfloor + n) \cdot K(m,n)$$

### 2.2 U-Net：医学图像分割
Ronneberger et al. (2015) 为医学图像设计。
- 编码器：逐步下采样，提取语义特征
- 解码器：逐步上采样，恢复空间分辨率
- 跳跃连接：把编码器的高分辨率特征拼接到解码器对应层

用 Mermaid 图展示 U-Net 的对称结构。

### 2.3 GAN：生成对抗框架
Goodfellow et al. (2014) 的极小极大博弈：
$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

训练不稳定的原因：
- 模式坍塌（Mode Collapse）：Generator 只生成少数几种样本
- 判别器太强：Generator 梯度消失
- 训练震荡：两者交替占优

### 2.4 DCGAN：让 GAN 稳定训练
Radford et al. (2015) 的工程实践：
- 用转置卷积做上采样（不用全连接层）
- BatchNorm 在 G 和 D 中都加（G 输出层和 D 输入层除外）
- 用 LeakyReLU（D）和 ReLU（G）
- 用 Adam 优化器，lr=0.0002, beta1=0.5

### 2.5 Progressive GAN：逐步增大分辨率
Karras et al. (2017) 的关键思想：
- 从 4×4 分辨率开始训练 G 和 D
- 训练稳定后，平滑地添加新层，分辨率翻倍
- 直到 1024×1024
- 每次添加新层时用 α 参数线性混合旧层和新层

### 2.6 Neural Style Transfer（简要）
Gatys et al. (2015) 的内容-风格分离：
- 内容损失：高层特征图的 MSE
- 风格损失：Gram 矩阵的 MSE
- 总损失 = α·内容损失 + β·风格损失

### 2.7 渐进式实现

Step 1 · U-Net 基本块:
```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """Double conv: Conv→BN→ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)
```

Step 2 · 完整 U-Net（编码器+解码器+跳跃连接）:
```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, base)          # 64
        self.enc2 = UNetBlock(base, base * 2)       # 128
        self.enc3 = UNetBlock(base * 2, base * 4)   # 256
        self.bottleneck = UNetBlock(base * 4, base * 8)  # 512
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = UNetBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = UNetBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = UNetBlock(base * 2, base)
        self.final = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
```

Step 3 · DCGAN Generator:
```python
class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

## 3. 工程要点
1. **GAN 训练不收敛** → D 太强 G 梯度消失
   处置：降低 D 的学习率或训练频率（每训练 G 一次，训练 D 两次改为 1:1）
2. **模式坍塌** → G 只生成少数几种输出
   处置：加入谱归一化（Spectral Normalization）或 Minibatch Discrimination
3. **分割边界模糊** → 上采样丢失细节
   处置：U-Net 跳跃连接是标配；必要时加 Deep Supervision
4. **GAN 评估困难** → 没有"准确率"这样的单一指标
   处置：FID（Fréchet Inception Distance）是主流，越低越好；IS（Inception Score）也可参考

> 你要记住：GAN 的排障优先级是 训练稳定性（loss 是否在震荡）→ 模式多样性（FID）→ 图像质量。

## 演进笔记
> 分割线从语义分割走向实例分割（Mask R-CNN, 2017）和全景分割。
> 生成线从 GAN 到 StyleGAN（2018-2019）的无条件高保真生成，再到 2020 年后扩散模型逐步取代 GAN。
> FID 后来成为图像生成评估的事实标准。
> 风格迁移后来被 CycleGAN 和 Neural Radiance Fields 继承发展。

→ 下一章：[轻量化架构 — 大模型很好，但手机装不下](../lightweight-vision/README.md)

---

**上一章**：[目标检测](../object-detection/README.md) | **下一章**：[轻量化架构](../lightweight-vision/README.md)
```

- [ ] **Step 2: Commit**

```bash
git add 01-Visual-Intelligence/segmentation-gan/README.md
git commit -m "content: add segmentation-gan module (Chinese)"
```

---

### Task 8: 创建 segmentation-gan/README_EN.md（英文）

**Files:**
- Create: `01-Visual-Intelligence/segmentation-gan/README_EN.md`

- [ ] **Step 1: 翻译中文 README 为英文**

完全翻译 Task 7 创建的中文版。要求：
- 顶部语言切换：`**[English](README_EN.md) | [中文](README.md)**`
- 标题: `# Understanding Pixels or Creating Them? — Segmentation & Generation (2015–2017)`
- 所有小节标题、正文、公式注释、代码注释翻译为英文
- 导航链接指向英文版

- [ ] **Step 2: Commit**

```bash
git add 01-Visual-Intelligence/segmentation-gan/README_EN.md
git commit -m "content: add segmentation-gan module (English)"
```

---

### Task 9: 创建 lightweight-vision/README.md（中文）

**Files:**
- Create: `01-Visual-Intelligence/lightweight-vision/README.md`

- [ ] **Step 1: 编写完整中文 README（~400-500 行）**

按照教学格式模板，内容结构如下：

```
# 大模型很好，但手机装不下 —— 轻量化架构（2016–2017）

## 这个问题从哪来
> ResNet-152 有 6000 万参数、需 11.3 GFLOPs。它能准确分类 ImageNet，
> 但部署到手机、无人机、嵌入式设备时，内存和算力都不够。
> 2016-2017 年，研究者开始追问：能不能用 1/10 甚至 1/50 的参数达到同等精度？
> SqueezeNet、MobileNet、SE-Net 从不同角度回答了这个问题。

## 学习目标
1. 深度可分离卷积为什么能把计算量降 8-9 倍？它的数学原理是什么？
2. SE-Net 的通道注意力机制如何让网络"学会关注重要通道"？
3. 这些轻量化策略如何影响了后续的 EfficientNet 和 NAS 方向？

## 1. 直觉
想象你有一个大型工具箱（标准卷积），里面什么工具都有但很重。
深度可分离卷积的做法是：把"看每个通道的空间模式"和"混合通道间信息"拆成两步。
就像先用手电筒逐个照（depthwise），再用混合器搅拌结果（pointwise）。
虽然步骤多了，但每步都很轻，总体快很多。

SE-Net 的直觉：给每个通道打个分——"这层的信息有用吗？"——有用的放大，没用的缩小。

> 你要记住：轻量化的三条路是①结构简化（MobileNet）、②通道筛选（SE-Net）、③架构搜索（NAS）。

## 2. 机制

### 2.1 SqueezeNet：1×1 卷积的压缩力
Iandola et al. (2016) — AlexNet 级精度，参数量仅 1.2MB（vs AlexNet 240MB）。
三个策略：
1. 用 1×1 卷积替换部分 3×3（squeeze 层：通道数先压后扩）
2. 减少输入通道数（3×3 前用 1×1 把通道压窄）
3. 延迟下采样（更大的特征图 → 更高精度）

Fire Module 结构：
- Squeeze：1×1 卷积，输出 s1 个通道
- Expand：1×1 卷积（e1 通道）+ 3×3 卷积（e3 通道），拼接
- 设计约束：s1 < e1 + e3（squeeze 比 expand 窄）

### 2.2 MobileNet：深度可分离卷积
Howard et al. (2017) — 把标准卷积拆成两步。

标准卷积计算量：$K \times K \times C_{in} \times C_{out} \times H \times W$
深度可分离卷积：
- Depthwise: $K \times K \times 1 \times C_{in} \times H \times W$
- Pointwise (1×1): $1 \times 1 \times C_{in} \times C_{out} \times H \times W$
- 计算量比：$\frac{1}{C_{out}} + \frac{1}{K^2} \approx \frac{1}{8} \text{~} \frac{1}{9}$（K=3, C_out=128~512）

宽度乘子 α（Width Multiplier）：统一缩放每层通道数（α ∈ {0.25, 0.5, 0.75, 1.0}）
分辨率乘子 ρ：降低输入分辨率（ρ ∈ {224, 192, 160, 128}）

### 2.3 SE-Net：通道注意力
Hu et al. (2017) — ILSVRC 2017 冠军。核心思想：让网络学到每层哪些通道更重要。

SE Block 流程：
1. Squeeze：全局平均池化 → (B, C, 1, 1) 每个通道一个标量
2. Excitation：FC → ReLU → FC → Sigmoid → (B, C) 通道权重
3. Scale：原特征图 × 通道权重

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z)), \quad z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_c(i,j)$$

关键设计：中间层降维比 r=16（W_1 把 C 压到 C/r，W_2 恢复到 C）

### 2.4 Capsule Networks（简要对比）
Hinton (2017) 的直觉：池化操作丢失了空间关系（"嘴巴在鼻子上方"）。
胶囊用动态路由替代最大池化，保留部分-整体的空间层级关系。
在 MNIST 上表现好，但扩展到大规模任务后未能替代 CNN。

### 2.5 渐进式实现

Step 1 · 深度可分离卷积：
```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = torch.relu(self.bn1(self.depthwise(x)))
        x = torch.relu(self.bn2(self.pointwise(x)))
        return x
```

Step 2 · SE Block：
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1, 1)
        return x * s
```

Step 3 · 计算量对比：
```python
def count_flops(conv_type, in_ch, out_ch, h, w, kernel=3):
    if conv_type == "standard":
        return kernel * kernel * in_ch * out_ch * h * w
    else:  # depthwise separable
        dw = kernel * kernel * in_ch * h * w
        pw = in_ch * out_ch * h * w
        return dw + pw

in_ch, out_ch, h, w = 64, 128, 56, 56
std_flops = count_flops("standard", in_ch, out_ch, h, w)
ds_flops  = count_flops("depthwise_separable", in_ch, out_ch, h, w)
print(f"标准卷积: {std_flops/1e6:.1f}M | 深度可分离: {ds_flops/1e6:.1f}M | 比率: {ds_flops/std_flops:.2%}")
# 输出: 标准卷积: 120.5M | 深度可分离: 26.8M | 比率: 22.26%
```

## 3. 工程要点
1. **深度可分离卷积精度损失** → 不是所有层都适合替换
   处置：浅层（边缘/纹理提取）用标准卷积，深层用深度可分离；或混合策略
2. **SE Block 开销** → 额外 FC 层增加参数
   处置：reduction ratio r=16 是经验最优，过小（r=2）参数多收益少
3. **量化部署** → 浮点模型转 INT8 精度下降
   处置：量化感知训练（QAT）或训练后量化（PTQ）+ 校准数据集
4. **延迟 vs 准确率的 Pareto 曲线** → 不能只看 FLOPs
   处置：FLOPs 低不等于实际推理快（内存访问模式也重要）；用目标设备实测

> 你要记住：轻量化的目标是"在目标设备上达到目标精度的最快模型"，不是"最少参数"。

## 演进笔记
> MobileNet → MobileNetV2/V3（倒残差结构、SE 模块融合）。
> SE-Net 的注意力思想后来被扩展到空间注意力（CBAM）和时间注意力。
> NAS（Neural Architecture Search）在 2018 年出现，用搜索替代手工设计。
> EfficientNet (2019) 用复合缩放（depth/width/resolution 同步）统一了轻量化方向。
> 轻量化路线最终被 ViT 的 patch embedding 和知识蒸馏部分取代——小模型从大模型蒸馏学习。

→ 下一阶段：[语言线 — 序列建模与 Transformer](../../02-Language-Transformers/README.md)

---

**上一章**：[分割与生成](../segmentation-gan/README.md) | **下一阶段**：[语言线](../../02-Language-Transformers/README.md)
```

- [ ] **Step 2: Commit**

```bash
git add 01-Visual-Intelligence/lightweight-vision/README.md
git commit -m "content: add lightweight-vision module (Chinese)"
```

---

### Task 10: 创建 lightweight-vision/README_EN.md（英文）

**Files:**
- Create: `01-Visual-Intelligence/lightweight-vision/README_EN.md`

- [ ] **Step 1: 翻译中文 README 为英文**

完全翻译 Task 9 创建的中文版。要求：
- 顶部语言切换：`**[English](README_EN.md) | [中文](README.md)**`
- 标题: `# Great Models, But They Don't Fit on Phones — Lightweight Architectures (2016–2017)`
- 导航链接指向英文版

- [ ] **Step 2: Commit**

```bash
git add 01-Visual-Intelligence/lightweight-vision/README_EN.md
git commit -m "content: add lightweight-vision module (English)"
```

---

## 代码练习（exercises/）

> 设计原则（来自 spec）：
> - 每模块 `exercises/` 目录，1-3 个 `.py` 文件
> - 渐进式：从最简实现到完整组件
> - 每个文件含注释说明 + 可运行代码 + assert 测试
> - 不依赖外部数据集，使用随机张量演示
> - 文件顶部注明依赖（torch 版本等）

---

### Task 11: 创建 training/exercises/training_loop.py

**Files:**
- Create: `01-Visual-Intelligence/training/exercises/training_loop.py`

- [ ] **Step 1: 创建 exercises 目录**

```bash
mkdir -p 01-Visual-Intelligence/training/exercises
```

- [ ] **Step 2: 编写 training_loop.py**

文件内容要求：

```python
"""
training_loop.py — 训练循环练习
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/training

练习目标:
  1. 实现最小训练循环（zero_grad → backward → step）
  2. 加入 BatchNorm + Dropout 正则化
  3. 加入梯度裁剪 + 学习率调度（Warmup + Cosine）
  4. 实现完整 epoch 循环 + 早停 + 检查点保存

所有数据使用随机张量，不依赖外部数据集。
"""
```

渐进式 4 步实现：

**Step 1 · 最小训练循环（验证三步顺序）**
- 定义 `nn.Sequential(Linear→ReLU→Linear)` 模型
- `SGD(lr=0.01)` + `CrossEntropyLoss`
- 随机 `(32, 16) → (32,)` 数据
- 断言: `loss.item()` 为有限值（`math.isfinite`）
- 打印 loss

**Step 2 · 加入 BN + Dropout**
- 定义 `RegularizedNet` 类：`Linear→BN1d→ReLU→Dropout(0.3)→Linear`
- 使用 `AdamW(weight_decay=1e-2)`
- 分别在 `model.train()` 和 `model.eval()` 模式下前向
- 断言: eval 模式输出 shape 正确

**Step 3 · 梯度裁剪 + LR 调度**
- `SequentialLR`: `LinearLR`（warmup 100 步）→ `CosineAnnealingLR`
- `clip_grad_norm_(max_norm=1.0)` 在 `backward` 后、`step` 前
- 断言: 调度后 lr < 初始 lr

**Step 4 · 完整 epoch 循环 + 早停**
- `run_epoch(model, loader, loss_fn, optimizer=None)` 函数
- `TensorDataset` 400 条随机数据，80/20 分 train/val
- `CosineAnnealingLR(T_max=20)` + patience=5 早停
- 保存最优模型到内存（`io.BytesIO`，不写磁盘）
- 断言: `best_val_acc > 0` 且为有限值

每个 Step 后用 `if __name__ == "__main__":` 块运行测试。

- [ ] **Step 3: Commit**

```bash
git add 01-Visual-Intelligence/training/exercises/training_loop.py
git commit -m "exercise: add training_loop.py for training module"
```

---

### Task 12: 创建 cnn-architectures/exercises/conv_basics.py + residual_block.py

**Files:**
- Create: `01-Visual-Intelligence/cnn-architectures/exercises/conv_basics.py`
- Create: `01-Visual-Intelligence/cnn-architectures/exercises/residual_block.py`

- [ ] **Step 1: 创建 exercises 目录**

```bash
mkdir -p 01-Visual-Intelligence/cnn-architectures/exercises
```

- [ ] **Step 2: 编写 conv_basics.py**

文件内容要求：

```python
"""
conv_basics.py — 卷积基础练习
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/cnn-architectures

练习目标:
  1. 手动实现 2D 卷积（无 nn.Conv2d），理解滑窗机制
  2. 验证感受野计算：3×3 两层 → 5×5 感受野
  3. 对比不同 kernel/stride/padding 对输出尺寸的影响
  4. 验证参数量公式：K²×C_in×C_out + C_out
"""
```

渐进式 4 步：

**Step 1 · 手动卷积（无 nn.Conv2d）**
- `manual_conv2d(x, kernel, stride=1, padding=0)` 函数
- 输入 `(1, 1, 5, 5)` + 3×3 kernel → 输出 `(1, 1, 3, 3)`
- 与 `nn.Conv2d(bias=False)` 对比，断言 `allclose`

**Step 2 · 输出尺寸公式**
- `output_size(input_size, kernel, stride, padding) = (input + 2*padding - kernel) // stride + 1`
- 测试 4 种组合：`(7,3,1,0)→5`, `(7,3,2,0)→3`, `(7,3,1,1)→7`, `(8,3,2,1)→4`
- 断言: 公式结果与 `nn.Conv2d` 实际输出尺寸一致

**Step 3 · 感受野计算**
- 两层 3×3 conv(stride=1, pad=1) → 感受野 = 5
- 三层 → 感受野 = 7
- 用零输入 + 中心为 1 的 kernel 验证感受野范围
- 断言: 非零输出只出现在感受野范围内

**Step 4 · 参数量对比**
- `count_parameters(model)` 函数
- 3×3 Conv(64→128): 73,856 参数 vs 1×1 Conv(64→128): 8,320 参数
- 5×5 Conv(64→128): 204,928 参数 vs 两层 3×3: 147,712 参数（感受野相同，参数更少）
- 断言: 两层 3×3 参数量 < 单层 5×5

- [ ] **Step 3: 编写 residual_block.py**

```python
"""
residual_block.py — 残差块练习
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/cnn-architectures

练习目标:
  1. 实现 BasicBlock（ResNet-18/34 的基本残差块）
  2. 实现 Bottleneck（ResNet-50/101/152 的瓶颈块）
  3. 验证残差连接能缓解梯度消失
  4. 实现通道变化时的 shortcut 投影
"""
```

渐进式 4 步：

**Step 1 · 最简残差块（无 BN，无下采样）**
- `SimpleResBlock(channels)`: `Conv3×3→ReLU→Conv3×3` + skip，最后 ReLU
- 输入 `(2, 64, 8, 8)` → 输出 `(2, 64, 8, 8)`
- 断言: 输出 shape 不变

**Step 2 · BasicBlock（完整版）**
- `Conv3×3→BN→ReLU→Conv3×3→BN` + shortcut + ReLU
- 支持通道变化（`in_channels != out_channels`）时的 1×1 conv shortcut
- 支持 `stride=2` 下采样
- 测试: `(2, 64, 32, 32)` → `(2, 128, 16, 16)`（通道翻倍，空间减半）
- 断言: shape 正确

**Step 3 · Bottleneck（ResNet-50+）**
- `1×1(压缩)→3×3→1×1(恢复)` 结构，expansion=4
- `in_channels=64 → bottleneck=64 → out_channels=256`
- shortcut 投影: `1×1 Conv(stride)` 当通道或尺寸不匹配时
- 测试: `(2, 64, 32, 32)` → `(2, 256, 16, 16)`
- 断言: shape 正确

**Step 4 · 梯度流验证**
- 对比有/无残差连接时，梯度在网络中间层的范数
- 10 层堆叠，输入 `(2, 64, 8, 8)`
- 断言: 有残差的梯度范数 > 无残差的梯度范数（至少 10×）

- [ ] **Step 4: Commit**

```bash
git add 01-Visual-Intelligence/cnn-architectures/exercises/
git commit -m "exercise: add conv_basics.py and residual_block.py for cnn-architectures"
```

---

### Task 13: 创建 object-detection/exercises/anchor_boxes.py + yolo_loss.py

**Files:**
- Create: `01-Visual-Intelligence/object-detection/exercises/anchor_boxes.py`
- Create: `01-Visual-Intelligence/object-detection/exercises/yolo_loss.py`

- [ ] **Step 1: 创建 exercises 目录**

```bash
mkdir -p 01-Visual-Intelligence/object-detection/exercises
```

- [ ] **Step 2: 编写 anchor_boxes.py**

```python
"""
anchor_boxes.py — Anchor 生成与 IoU 计算
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/object-detection

练习目标:
  1. 实现 IoU（交并比）计算
  2. 实现 Anchor 网格生成
  3. 实现 NMS（非极大值抑制）
  4. 实现偏移量编码/解码（YOLO/SSD 风格）
"""
```

渐进式 4 步：

**Step 1 · IoU 计算**
- `compute_iou(boxes1, boxes2) → (N, M)` 矩阵
- boxes 格式: `(N, 4)` xyxy
- 交集 = `max(x1) min(x2) × max(y1) min(y2).clamp(min=0)`
- 并集 = area1 + area2 - 交集
- 测试: 两个完全重合框 IoU = 1.0；两个不重叠框 IoU = 0.0
- 断言: 形状 `(5, 3)` 输入 → `(5, 3)` IoU 矩阵

**Step 2 · Anchor 网格生成**
- `generate_anchors(feature_size, stride, scales, ratios) → (K, 4)`
- 在特征图每个位置生成 |scales|×|ratios| 个 anchor
- 中心坐标: `((x + 0.5) × stride, (y + 0.5) × stride)`
- 宽高: `w = s / sqrt(r)`, `h = s × sqrt(r)`
- 测试: `feature=13, stride=32, scales=[128,256], ratios=[0.5,1,2]` → `13×13×6 = 1014` 个 anchor
- 断言: anchor 数量正确

**Step 3 · NMS**
- `nms(boxes, scores, iou_threshold=0.5) → keep_indices`
- 按 score 降序排列，贪心剔除 IoU > threshold 的重叠框
- 测试: 6 个框中有 3 对高度重叠 → NMS 保留 3 个
- 断言: `len(keep) == 3`

**Step 4 · 偏移量编码/解码**
- `encode_boxes(gt, anchors) → offsets`: `tx = (gx - ax) / aw`, `ty = (gy - ay) / ah`, `tw = log(gw / aw)`, `th = log(gh / ah)`
- `decode_boxes(offsets, anchors) → pred`: 逆运算
- 断言: `decode(encode(gt, anchors), anchors) ≈ gt`（allclose）

- [ ] **Step 3: 编写 yolo_loss.py**

```python
"""
yolo_loss.py — YOLO 损失函数
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/object-detection

练习目标:
  1. 实现 YOLO v1 的坐标损失（宽高取平方根）
  2. 实现置信度损失（有物体/无物体分开加权）
  3. 实现分类损失
  4. 组合为完整 YOLO Loss
"""
```

渐进式 4 步：

**Step 1 · 坐标损失**
- `coord_loss(pred_xy, pred_wh, target_xy, target_wh, obj_mask) → loss`
- 只计算有物体的网格位置（`obj_mask`）
- 宽高损失用 `sqrt(pred_wh)` vs `sqrt(target_wh)`
- 测试: 完美预测 → loss ≈ 0
- 断言: loss 标量且有限

**Step 2 · 置信度损失**
- `conf_loss(pred_conf, target_conf, obj_mask, lambda_noobj=0.5) → loss`
- 有物体位置: `BCE(pred, 1)`，权重 `λ_coord`
- 无物体位置: `BCE(pred, 0)`，权重 `λ_noobj`
- 断言: 无物体项权重 < 有物体项权重

**Step 3 · 分类损失**
- `class_loss(pred_class, target_class, obj_mask) → loss`
- 只在有物体的位置计算 `CrossEntropy`
- target_class: `(B, S, S)` int 类型
- 断言: 形状正确

**Step 4 · 完整 YOLO Loss**
- `YOLOLoss(S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5)`
- 组合三部分: `λ_coord × coord + conf_obj + λ_noobj × conf_noobj + class`
- 输入: `(B, S, S, B×5+C)` 预测张量 + 同形目标张量
- 测试: 随机预测 loss > 0；完美预测 loss ≈ 0
- 断言: loss 为正标量

- [ ] **Step 4: Commit**

```bash
git add 01-Visual-Intelligence/object-detection/exercises/
git commit -m "exercise: add anchor_boxes.py and yolo_loss.py for object-detection"
```

---

### Task 14: 创建 segmentation-gan/exercises/unet.py + dcgan_generator.py

**Files:**
- Create: `01-Visual-Intelligence/segmentation-gan/exercises/unet.py`
- Create: `01-Visual-Intelligence/segmentation-gan/exercises/dcgan_generator.py`

- [ ] **Step 1: 创建 exercises 目录**

```bash
mkdir -p 01-Visual-Intelligence/segmentation-gan/exercises
```

- [ ] **Step 2: 编写 unet.py**

```python
"""
unet.py — U-Net 编码器-解码器实现
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/segmentation-gan

练习目标:
  1. 实现 UNetBlock（Double Conv: Conv→BN→ReLU × 2）
  2. 实现完整 U-Net（编码器+瓶颈+解码器+跳跃连接）
  3. 验证跳跃连接对分割边界的影响
  4. 测量参数量和 FLOPs
"""
```

渐进式 4 步：

**Step 1 · UNetBlock**
- `Conv2d(in, out, 3, pad=1)→BN→ReLU→Conv2d(out, out, 3, pad=1)→BN→ReLU`
- 测试: `(2, 64, 16, 16)` → `(2, 128, 16, 16)`（通道变，尺寸不变）
- 断言: shape 正确

**Step 2 · 编码器 + 解码器**
- 编码器: 4 级 `UNetBlock + MaxPool2d(2)`，通道 1→64→128→256→512
- 瓶颈: `UNetBlock(512, 1024)`
- 解码器: 4 级 `ConvTranspose2d(stride=2) + cat(skip) + UNetBlock`
- 测试: `(2, 1, 128, 128)` → `(2, 1, 128, 128)`
- 断言: 输出尺寸 == 输入尺寸

**Step 3 · 跳跃连接对比**
- 创建 `UNetNoSkip`（去掉跳跃连接，decoder 输入只有上采样结果）
- 对同一输入，对比两个模型的中间特征图 L2 范数
- 断言: 有跳跃连接的解码器特征范数更大（信息更丰富）

**Step 4 · 参数量统计**
- `count_parameters(model)` → 总参数量
- U-Net(base=64): ~7.7M 参数
- 打印层级参数分布
- 断言: 参数量在预期范围内（±10%）

- [ ] **Step 3: 编写 dcgan_generator.py**

```python
"""
dcgan_generator.py — DCGAN 生成器实现
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/segmentation-gan

练习目标:
  1. 实现单个转置卷积上采样块
  2. 实现完整 DCGAN Generator
  3. 实现 DCGAN Discriminator（镜像结构）
  4. 验证 GAN 训练循环的 loss 动态
"""
```

渐进式 4 步：

**Step 1 · 转置卷积上采样块**
- `UpBlock(in_ch, out_ch)`: `ConvTranspose2d(kernel=4, stride=2, padding=1)→BN→ReLU`
- 测试: `(2, 512, 2, 2)` → `(2, 256, 4, 4)`（空间尺寸翻倍）
- 断言: shape 正确

**Step 2 · DCGAN Generator**
- `DCGANGenerator(z_dim=100, base_ch=64, out_ch=3)`
- 结构: `z → (z_dim, 1, 1) → UpBlock × 4 → ConvTranspose2d(3) → Tanh`
- 输出: `(B, 3, 64, 64)`
- 测试: 随机噪声 `(2, 100)` → `(2, 3, 64, 64)`
- 断言: 输出值在 `[-1, 1]` 范围内（Tanh）

**Step 3 · DCGAN Discriminator**
- `DCGANDiscriminator(in_ch=3, base_ch=64)`
- 镜像 Generator: `Conv2d(stride=2)→LeakyReLU(0.2)` → 最后 `Sigmoid`
- 输入 `(B, 3, 64, 64)` → 输出 `(B, 1)` 真假概率
- 断言: 输出值在 `[0, 1]`

**Step 4 · GAN 训练循环验证**
- 1 步 G 训练: `z → G(z) → D(G(z)) → BCE → G_loss`
- 1 步 D 训练: `D(real) → BCE(real=1); D(G(z).detach()) → BCE(real=0)`
- 断言: 两个 loss 均为正有限值
- 打印 G_loss 和 D_loss

- [ ] **Step 4: Commit**

```bash
git add 01-Visual-Intelligence/segmentation-gan/exercises/
git commit -m "exercise: add unet.py and dcgan_generator.py for segmentation-gan"
```

---

### Task 15: 创建 lightweight-vision/exercises/depthwise_separable.py

**Files:**
- Create: `01-Visual-Intelligence/lightweight-vision/exercises/depthwise_separable.py`

- [ ] **Step 1: 创建 exercises 目录**

```bash
mkdir -p 01-Visual-Intelligence/lightweight-vision/exercises
```

- [ ] **Step 2: 编写 depthwise_separable.py**

```python
"""
depthwise_separable.py — 深度可分离卷积 vs 标准卷积对比
依赖: torch >= 2.0
所属模块: 01-Visual-Intelligence/lightweight-vision

练习目标:
  1. 实现深度可分离卷积（Depthwise + Pointwise）
  2. 实现标准卷积的 FLOPs 计算
  3. 实现深度可分离卷积的 FLOPs 计算
  4. 实现 SE Block（通道注意力）
  5. 参数量和 FLOPs 对比验证
"""
```

渐进式 5 步：

**Step 1 · DepthwiseSeparableConv 模块**
- `DepthwiseSeparableConv(in_ch, out_ch, stride=1)`
- `Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)` (depthwise)
- `BatchNorm2d(in_ch) → ReLU`
- `Conv2d(in_ch, out_ch, 1, bias=False)` (pointwise)
- `BatchNorm2d(out_ch) → ReLU`
- 测试: `(2, 64, 32, 32)` → `(2, 128, 32, 32)` (stride=1)
- 测试: `(2, 64, 32, 32)` → `(2, 128, 16, 16)` (stride=2)
- 断言: shape 正确

**Step 2 · 标准卷积 FLOPs**
- `count_standard_flops(in_ch, out_ch, h, w, kernel=3) → int`
- 公式: `kernel² × in_ch × out_ch × h × w`
- 测试: `(64, 128, 56, 56)` → `120,422,400`

**Step 3 · 深度可分离 FLOPs**
- `count_ds_flops(in_ch, out_ch, h, w, kernel=3) → int`
- depthwise: `kernel² × in_ch × h × w`
- pointwise: `in_ch × out_ch × h × w`
- 测试: `(64, 128, 56, 56)` → `26,705,920`
- 断言: ds_flops / std_flops ≈ 22% (约 1/8 ~ 1/9 的计算量)

**Step 4 · SE Block**
- `SEBlock(channels, reduction=16)`
- `AdaptiveAvgPool2d(1) → Linear(C, C//r) → ReLU → Linear(C//r, C) → Sigmoid`
- `forward: x * s`（通道加权）
- 测试: `(2, 64, 16, 16)` → `(2, 64, 16, 16)`（shape 不变）
- 断言: 输出各通道的均值不全相等（注意力生效）

**Step 5 · 综合对比**
- 打印表格对比 4 种配置:
  | 配置 | 参数量 | FLOPs | 比率 |
  | 标准卷积 64→128 | ... | ... | 100% |
  | 深度可分离 64→128 | ... | ... | ~22% |
  | 标准卷积 128→256 | ... | ... | 100% |
  | 深度可分离 128→256 | ... | ... | ~22% |
- 断言: 所有深度可分离配置的 FLOPs < 标准卷积的 30%

- [ ] **Step 3: Commit**

```bash
git add 01-Visual-Intelligence/lightweight-vision/exercises/depthwise_separable.py
git commit -m "exercise: add depthwise_separable.py for lightweight-vision"
```

---

## 联动更新

---

### Task 16: 验证时间线同步

**Files:**
- Check: `00-Timeline/README.md`

- [ ] **Step 1: 检查时间线条目覆盖**

对照 spec 中列出的时间线条目，确认所有新模块内容在 `00-Timeline/README.md` 中已有对应条目：

| 模块 | 需覆盖的条目 |
|------|------------|
| object-detection | Faster R-CNN (2015)、YOLO v1 (2015)、SSD (2016)、Focal Loss/RetinaNet (2017) |
| segmentation-gan | U-Net (2015)、DCGAN (2015)、Progressive GAN (2017)、Neural Style Transfer (2015) |
| lightweight-vision | SqueezeNet (2016)、MobileNet (2017)、SE-Net (2017)、Capsule Networks (2017) |

- [ ] **Step 2: 如有缺失条目，补充到时间线**

如果 `00-Timeline/README.md` 中缺少任何上述条目，按现有格式添加。

- [ ] **Step 3: 验证模块→时间线反向链接**

确认每个新模块 README 的"演进笔记"或正文中的时间线引用与 `00-Timeline/` 的内容一致。如有不一致，更新模块 README。

- [ ] **Step 4: Commit（如有改动）**

```bash
git add 00-Timeline/README.md
git commit -m "sync: align timeline entries with new visual-intelligence modules"
```

（如果时间线无需修改，跳过此步）

---

### Task 17: 最终验证与导航链接检查

**Files:**
- 所有新建和修改的文件

- [ ] **Step 1: 检查所有相对链接有效性**

逐个验证以下链接类别：

1. **章节→模块链接**: `01-Visual-Intelligence/README.md` 中 6 个模块链接路径正确
2. **模块间前后导航**: 每个模块底部 `**上一章**` / `**下一章**` 链接指向正确文件
3. **跨章节链接**: `lightweight-vision → 02-Language-Transformers` 路径正确
4. **前置引用链接**: 模块中引用 `00-Prerequisites/` 的路径正确
5. **英文版链接**: 所有 `README_EN.md` 中的链接指向英文版而非中文版
6. **语言切换链接**: 每对 `README.md` / `README_EN.md` 顶部双向链接正确

- [ ] **Step 2: 检查文件结构完整性**

确认最终文件树与 spec 的目标结构完全一致：

```
01-Visual-Intelligence/
├── README.md              ✓ 存在且已修改
├── README_EN.md           ✓ 存在（新建）
├── training/
│   ├── README.md          ✓ 存在（未改）
│   ├── README_EN.md       ✓ 存在（已重写）
│   └── exercises/
│       └── training_loop.py ✓ 存在（新建）
├── cnn-architectures/
│   ├── README.md          ✓ 存在（未改）
│   ├── README_EN.md       ✓ 存在（未改）
│   └── exercises/
│       ├── conv_basics.py ✓ 存在（新建）
│       └── residual_block.py ✓ 存在（新建）
├── object-detection/      ✓ 存在（新建目录）
│   ├── README.md
│   ├── README_EN.md
│   └── exercises/
│       ├── anchor_boxes.py
│       └── yolo_loss.py
├── segmentation-gan/      ✓ 存在（新建目录）
│   ├── README.md
│   ├── README_EN.md
│   └── exercises/
│       ├── unet.py
│       └── dcgan_generator.py
├── lightweight-vision/    ✓ 存在（新建目录）
│   ├── README.md
│   ├── README_EN.md
│   └── exercises/
│       └── depthwise_separable.py
└── sequence-models/       ✓ 存在
    ├── README.md          ✓ 已修改
    └── README_EN.md       ✓ 已重写
```

- [ ] **Step 3: 检查 Markdown 渲染**

目视检查所有新建/修改的 README：
- LaTeX 公式有 `$$...$$` 或 `$...$` 分隔符
- Mermaid 代码块有 ` ```mermaid ` 标记
- 代码块有正确的语言标记（`python`）
- 表格分隔符完整

---

## 任务依赖关系

```
Task 1 (导航顺序) ─────────────────────────────────┐
Task 2 (章节 README_EN) ← 依赖 Task 1 完成          │
Task 3 (training EN)     ← 独立                      │
Task 4 (sequence-models) ← 独立                      │
Task 5 (detection CN)    ← 独立                      │
Task 6 (detection EN)    ← 依赖 Task 5               │
Task 7 (seg/gan CN)      ← 独立                      │
Task 8 (seg/gan EN)      ← 依赖 Task 7               │
Task 9 (lightweight CN)  ← 独立                      │
Task 10 (lightweight EN) ← 依赖 Task 9               │
Task 11 (training ex)    ← 独立                      │
Task 12 (cnn ex)         ← 独立                      │
Task 13 (detection ex)   ← 依赖 Task 5 目录存在       │
Task 14 (seg/gan ex)     ← 依赖 Task 7 目录存在       │
Task 15 (lightweight ex) ← 依赖 Task 9 目录存在       │
Task 16 (时间线同步)     ← 依赖 Task 5,7,9           │
Task 17 (最终验证)       ← 依赖所有其他 Task          │
```

可并行分组：
- **组 A**: Task 1, 3, 4, 11, 12（无依赖，可同时开始）
- **组 B**: Task 2（等 Task 1）, Task 5, 7, 9（无依赖）
- **组 C**: Task 6（等 5）, Task 8（等 7）, Task 10（等 9）, Task 13, 14, 15（等对应目录）
- **组 D**: Task 16（等 B 组完成）
- **组 E**: Task 17（等所有完成）

