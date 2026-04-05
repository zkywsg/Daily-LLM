# 01-Visual-Intelligence 章节扩展与优化设计

> 日期: 2026-04-05
> 状态: 待实现

## 背景

`01-Visual-Intelligence/` 当前仅有 3 个子模块（training、cnn-architectures、sequence-models），但 `00-Timeline/README.md` 中已分配给该章节的 2012-2017 视觉线索条目远不止于此。目标检测（Faster R-CNN、YOLO、SSD、RetinaNet）、图像分割（U-Net）、生成模型（GAN/DCGAN/Progressive GAN）、轻量化架构（MobileNet、SqueezeNet）等重要主题均缺少对应模块。

此外，现有模块存在若干结构问题：导航顺序不一致、英文翻译过时、sequence-models 迁移不完整、缺少代码练习。

## 目标

1. **内容补充**：新增 3 个主题模块，覆盖时间线中缺失的视觉方向
2. **结构修复**：统一导航顺序、同步英文翻译、完善迁移提示、修复断链
3. **代码练习**：为每个模块（现有+新增）添加 PyTorch 代码练习文件

## 约束

- 时间范围严格限定 2012-2017
- 新模块沿用现有叙事式教学格式（问题起源→直觉→机制→演进→代码→工程要点→演进笔记）
- 中文优先，英文翻译同步
- 代码练习为 `.py` 文件，不依赖外部数据集

## 方案选择

### 方案 A（已采纳）：3 个新模块，按主题分组

| 新模块 | 覆盖内容 |
|--------|---------|
| `object-detection/` | Faster R-CNN → YOLO → SSD → RetinaNet |
| `segmentation-gan/` | FCN → U-Net → GAN → DCGAN → Progressive GAN |
| `lightweight-vision/` | SqueezeNet → MobileNet → SE-Net + Capsule Networks |

理由：模块粒度适中，主题内聚。segmentation 和 GAN 共享编码器-解码器架构主线；轻量化模块统一覆盖 2016-2017 效率优化方向。

### 方案 B（未采纳）：5 个更细粒度模块

将方案 A 中的 segmentation/gan 拆开，并单独设 vision-applications 模块。问题：部分模块内容偏薄，总数达 8 个，容易碎片化。

## 目标文件结构

```
01-Visual-Intelligence/
├── README.md                      (更新：统一导航顺序)
├── README_EN.md                   (新增：章节级英文概览)
├── training/
│   ├── README.md                  (已有)
│   ├── README_EN.md               (重写：匹配叙事式中文版)
│   └── exercises/
│       └── training_loop.py       (新增)
├── cnn-architectures/
│   ├── README.md                  (已有)
│   ├── README_EN.md               (已有)
│   └── exercises/
│       ├── conv_basics.py         (新增)
│       └── residual_block.py      (新增)
├── object-detection/              (新增模块)
│   ├── README.md
│   ├── README_EN.md
│   └── exercises/
│       ├── anchor_boxes.py
│       └── yolo_loss.py
├── segmentation-gan/              (新增模块)
│   ├── README.md
│   ├── README_EN.md
│   └── exercises/
│       ├── unet.py
│       └── dcgan_generator.py
├── lightweight-vision/            (新增模块)
│   ├── README.md
│   ├── README_EN.md
│   └── exercises/
│       └── depthwise_separable.py
└── sequence-models/               (迁移残留)
    ├── README.md                  (更新迁移提示)
    └── README_EN.md               (重写为迁移提示，修复断链)
```

**阅读顺序：** Training → CNN Architectures → Object Detection → Segmentation & GAN → Lightweight Vision → (过渡到 02-Language-Transformers)

## 新模块详细设计

### 模块：object-detection/（目标检测）

**叙事主线：** 从"分类一张图"到"找到图里所有东西"

| 小节 | 内容 |
|------|------|
| 问题起源 | 图像分类只回答"是什么"，但现实需要"在哪里、有几个" |
| 滑窗与区域提案 | R-CNN → Fast R-CNN → Faster R-CNN 三代演进，RPN 关键洞察 |
| 单阶段检测 | YOLO v1：把检测变成回归问题，速度与精度的权衡 |
| 多尺度检测 | SSD 多尺度特征图检测，为什么不同大小物体需要不同层 |
| 类别不平衡 | RetinaNet / Focal Loss：正负样本极端不平衡问题及其数学解法 |
| 工程要点 | NMS、mAP 评估指标、Anchor 设计 |
| 演进笔记 | 锚框到 anchor-free 的方向，指向后续发展 |

**时间线条目覆盖：** Faster R-CNN (2015)、YOLO v1 (2015)、SSD (2016)、Focal Loss/RetinaNet (2017)

**代码练习：**
- `anchor_boxes.py` — Anchor 生成与 IoU 计算
- `yolo_loss.py` — YOLO 损失函数实现

### 模块：segmentation-gan/（分割与生成）

**叙事主线：** 理解像素 vs 创造像素 — 编码器-解码器的两副面孔

| 小节 | 内容 |
|------|------|
| 问题起源 | 分类和检测都输出矩形框，现实需要像素级精确划分 |
| 语义分割 | FCN：全卷积化，分类网络变成逐像素分类器 |
| U-Net | 编码器-解码器 + 跳跃连接，医学图像分割经典架构 |
| 转向生成 | 编码器-解码器能"还原"空间信息 → 能否"创造"新图像 |
| GAN | Generator vs Discriminator 博弈直觉 |
| DCGAN | 卷积 + GAN 工程实践，稳定训练关键技巧 |
| Progressive GAN | 逐步增大分辨率 4×4 到 1024×1024 |
| 风格迁移 | Neural Style Transfer 内容与风格分离（简要） |
| 演进笔记 | GAN→StyleGAN，分割→实例分割(Mask R-CNN)，指向 03 章节扩散模型 |

**时间线条目覆盖：** U-Net (2015)、DCGAN (2015)、Progressive GAN (2017)、Neural Style Transfer (2015)

**代码练习：**
- `unet.py` — U-Net 编码器-解码器实现
- `dcgan_generator.py` — DCGAN 生成器实现

### 模块：lightweight-vision/（轻量化架构）

**叙事主线：** 大模型很好，但手机装不下

| 小节 | 内容 |
|------|------|
| 问题起源 | ResNet-152 有 6000 万参数，移动端部署不现实 |
| SqueezeNet | AlexNet 级精度 1/50 参数量 — 1×1 squeeze 压缩策略 |
| MobileNet | 深度可分离卷积：depthwise + pointwise，计算量降 8-9 倍 |
| SE-Net | 通道注意力：让网络学会关注哪些通道，ILSVRC 2017 冠军 |
| Capsule Networks | Hinton 直觉：池化丢失空间关系，动态路由替代（对比视角，简要） |
| 演进笔记 | EfficientNet、NAS → ViT 出现意味着 CNN 路线收尾 |

**时间线条目覆盖：** SqueezeNet (2016)、MobileNet (2017)、SE-Net (2017)、Capsule Networks (2017)

**代码练习：**
- `depthwise_separable.py` — 深度可分离卷积 vs 标准卷积对比

## 结构修复清单

| 修复项 | 操作 |
|--------|------|
| 章节导航顺序 | README.md 模块列表改为 Training→CNN→Detection→Seg/GAN→Lightweight→Sequence |
| 章节级 README_EN.md | 新增英文版章节概览 |
| training/README_EN.md | 重写为叙事式教学格式，匹配中文版 |
| sequence-models/README_EN.md | 重写为迁移提示页，匹配中文版 |
| sequence-models 断链 | 修复指向 03-NLP-Transformers 的链接为正确路径 |
| 所有模块导航链接 | 统一为 Training→CNN→Detection→Seg/GAN→Lightweight 顺序 |

## 代码练习设计原则

- 每模块 `exercises/` 目录，1-3 个 `.py` 文件
- 渐进式：从最简实现到完整组件
- 每个文件含注释说明 + 可运行代码 + assert 测试
- 不依赖外部数据集，使用随机张量演示
- 文件顶部注明依赖（torch 版本等）

## 教学格式模板（与现有模块统一）

1. **问题起源** — "这个技术要解决什么痛点？"
2. **直觉解释** — 不用公式先说清楚
3. **数学机制** — 核心公式与推导
4. **架构演进** — 逐代讲，每代解决上代什么问题
5. **PyTorch 代码示例** — 嵌入 README 内
6. **工程要点 / 常见坑** — 实践经验
7. **演进笔记** — 承前启后，连接上下游章节

## 联动更新

- `00-Timeline/README.md` — 确认新模块覆盖的时间线条目无需新增或修改
- `01-Visual-Intelligence/README.md` — 更新模块列表和导航顺序
- 各模块底部导航链接 — 统一更新
- `sequence-models/` — 中英文版迁移提示同步

## 实现优先级

所有工作并行推进，但建议的审查顺序：

1. 结构修复（导航、翻译、迁移、断链）— 基础设施
2. 新模块 README.md（中文）— 内容核心
3. 新模块 README_EN.md（英文）— 翻译
4. 代码练习（exercises/）— 实践补充
5. 联动更新（章节概览、导航链接、时间线同步）
