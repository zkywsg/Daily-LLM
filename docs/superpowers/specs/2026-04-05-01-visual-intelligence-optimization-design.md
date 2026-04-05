# Phase 01 · Visual Intelligence 全面优化设计方案

**日期**: 2026-04-05
**范围**: 01-Visual-Intelligence 章节的结构性修复、内容补全、模块重组与新增

---

## 一、变更总览

| 类型 | 变更 | 优先级 |
|------|------|--------|
| 删除 | 移除 `sequence-models/` 模块（内容已在 02-Language-Transformers） | 高 |
| 新增 | 添加 `gan-advanced/` 模块（CycleGAN、Pix2Pix、StyleGAN、cGAN） | 高 |
| 修复 | 补全 `segmentation-gan/README_EN.md` 缺失的 4 个章节 | 高 |
| 修复 | 主 `README_EN.md` 中指向 `00-Timeline/README_EN.md` 的断链 | 高 |
| 修复 | 主 `README.md` 添加语言切换标头 | 中 |
| 更新 | 主 README.md/README_EN.md 模块列表和导航 | 高 |
| 更新 | 各子模块的 prev/next 导航链接 | 高 |

---

## 二、删除 sequence-models 模块

### 理由

- RNN/LSTM/GRU/Seq2Seq 的完整内容已在 `02-Language-Transformers/recurrent-networks/`
- sequence-models 目前是跳转入口，实际读者已从 02 获取内容
- 01 应保持纯视觉定位（2012–2017），语言模型内容归 02
- 删除后为新的 GAN 进阶模块腾出空间

### 执行

- 删除 `01-Visual-Intelligence/sequence-models/` 目录
- 从主 `README.md` 和 `README_EN.md` 中移除序列模型章节条目
- 从时间线表中移除相关条目（如有）

---

## 三、新增 gan-advanced 模块

### 定位

在 `segmentation-gan/`（基础 GAN + DCGAN + Progressive GAN）之后，GAN 进阶模块覆盖 2017–2018 年的核心 GAN 变体，填补"基础 GAN → 扩散模型"之间的知识空白。

### 目录结构

```
gan-advanced/
├── README.md          （中文，主文档）
├── README_EN.md       （英文翻译）
└── exercises/
    ├── cyclegan_cycle_consistency.py
    ├── conditional_gan.py
    └── stylegan_mapping_network.py
```

### 内容大纲

遵循现有模块的标准结构：直觉 → 机制 → 渐进式代码实现 → 工程要点 → 演进笔记

#### 1. 直觉
- 为什么需要 GAN 变体：原始 GAN 的三大痛点（训练不稳定、模式坍塌、可控性差）
- 从"无条件生成"到"条件控制"再到"无配对转换"的思路演进

#### 2. 机制

**2.1 条件 GAN（cGAN）— Mirza & Osindero, 2014**
- 在 G 和 D 中同时注入条件信息（类别标签、文本嵌入等）
- 损失函数的条件化扩展
- 应用：文本到图像、类别条件生成

**2.2 Pix2Pix — Isola et al., 2017**
- 配对图像到图像翻译（edges→photo, day→night, map→aerial）
- U-Net Generator + PatchGAN Discriminator
- L1 重建损失 + 对抗损失的组合
- PatchGAN：判别器只判断图像局部块的真伪

**2.3 CycleGAN — Zhu et al., 2017**
- 无配对图像翻译的核心创新
- Cycle Consistency Loss：A→B→A 应该回到原图
- 两个 Generator + 两个 Discriminator 的对称架构
- 应用：马↔斑马、照片↔油画、季节转换

**2.4 StyleGAN — Karras et al., 2018-2019**
- Mapping Network：将 z 映射到中间潜空间 W
- Adaptive Instance Normalization（AdaIN）注入风格
- 噪声注入层控制随机细节
- 风格混合与潜空间插值
- StyleGAN v1→v2→v3 的演进概述

#### 3. 渐进式代码实现
- Step 1: cGAN Generator + Discriminator
- Step 2: PatchGAN 判别器
- Step 3: CycleGAN 的 Cycle Consistency 损失
- Step 4: StyleGAN 的 Mapping Network + AdaIN

#### 4. 工程要点
- 4.1 CycleGAN 训练不稳定 → 身份损失（Identity Loss）的引入
- 4.2 Pix2Pix 模糊输出 → 减少 L1 权重、使用 PatchGAN
- 4.3 StyleGAN 渐进训练 → 稳定性正则化
- 4.4 GAN 评估进阶：FID 的实际计算与陷阱

#### 5. 关键论文与时间线

| 年份 | 论文 | 核心贡献 |
|------|------|----------|
| 2014 | Mirza & Osindero, *Conditional GAN* | 条件化生成控制 |
| 2017 | Isola et al., *Pix2Pix* | 配对图像翻译 + PatchGAN |
| 2017 | Zhu et al., *CycleGAN* | 无配对翻译 + 循环一致性 |
| 2018 | Karras et al., *StyleGAN* | 潜空间分层风格控制 |
| 2020 | Karras et al., *StyleGAN 2* | 消除伪影、路径正则化 |
| 2021 | Karras et al., *StyleGAN 3* | 消除纹理粘连 |

#### 6. 概念速查表

#### 7. 练习与思考
- 基础理解（3 题）
- 动手实验（3 题）
- 进阶思考（2 题）

#### 8. 演进笔记
- GAN → 扩散模型的过渡（简要预告）
- GAN 在 2024 年后的定位

### 导航关系

- **上一章**: `segmentation-gan/`（分割与生成 → GAN 进阶）
- **下一章**: `lightweight-vision/`（轻量化架构）

---

## 四、修复 segmentation-gan/README_EN.md

### 缺失章节

中文版（634 行）vs 英文版（459 行），缺失约 174 行：

| 缺失章节 | 中文行范围 | 内容 |
|----------|-----------|------|
| Section 3.5 | 525–530 | 分割类别不平衡 |
| Section 3.6 | 533–540 | GPU 显存管理 |
| Section 6 | 572–589 | 分割 vs 生成对比分析表 |
| Section 7 | 592–615 | 练习与思考 |

### 执行

- 将缺失的 4 个章节翻译为英文并插入 README_EN.md 对应位置
- 确保英文版的章节编号和结构与中文版一致

---

## 五、修复断链与语言切换

### 5.1 主 README_EN.md 断链修复

当前链接：`[00-Timeline](../00-Timeline/)` — 此链接本身有效（指向目录）。
但主 README_EN.md 内部引用了 `../00-Timeline/README_EN.md`，该文件不存在。

**方案**: 修改链接指向 `../00-Timeline/README.md`（中文版），因为时间线暂无英文版。添加注释说明。

### 5.2 主 README.md 语言切换

在文件开头添加标准语言切换标头：

```markdown
**[English](README_EN.md) | [中文](README.md)**
```

---

## 六、更新模块列表与导航

### 6.1 主 README.md 更新

移除序列模型条目，添加 GAN 进阶条目：

```
## 本阶段内容

### [训练与优化](training/README.md)
...

### [CNN 架构](cnn-architectures/README.md)
...

### [目标检测](object-detection/README.md)
...

### [分割与生成](segmentation-gan/README.md)
...

### [GAN 进阶](gan-advanced/README.md)          ← 新增
条件 GAN、Pix2Pix、CycleGAN、StyleGAN
- 从无条件生成到条件控制
- 配对与无配对图像翻译
- 潜空间分层风格控制

### [轻量化架构](lightweight-vision/README.md)
...
```

### 6.2 主 README_EN.md 同步更新

英文版镜像更新，移除 Sequence Models，添加 Advanced GAN。

### 6.3 子模块导航链接更新

需要更新的 prev/next 链接：

| 模块 | 变更 |
|------|------|
| `segmentation-gan/README.md` | next: `../gan-advanced/README.md`（原 `../lightweight-vision/`） |
| `segmentation-gan/README_EN.md` | next: `../gan-advanced/README_EN.md` |
| `gan-advanced/README.md` | prev: `../segmentation-gan/`，next: `../lightweight-vision/` |
| `gan-advanced/README_EN.md` | prev: `../segmentation-gan/`，next: `../lightweight-vision/` |
| `lightweight-vision/README.md` | prev: `../gan-advanced/`（原 `../segmentation-gan/`） |
| `lightweight-vision/README_EN.md` | prev: `../gan-advanced/` |

### 6.4 时间线更新

在 00-Timeline/README.md 中检查是否需要补充：
- 2017: CycleGAN（无配对图像翻译）
- 2018: StyleGAN（高保真面部生成）

---

## 七、执行顺序

1. 删除 `sequence-models/` 目录
2. 创建 `gan-advanced/` 目录和内容（README.md + README_EN.md + exercises）
3. 补全 `segmentation-gan/README_EN.md` 缺失章节
4. 修复主 `README.md`（添加语言切换、更新模块列表）
5. 修复主 `README_EN.md`（修复断链、更新模块列表）
6. 更新所有子模块的 prev/next 导航链接
7. 检查并更新 00-Timeline（如需要）
8. 提交

---

## 八、验收标准

- [ ] `sequence-models/` 目录已删除
- [ ] `gan-advanced/` 目录完整（README.md + README_EN.md + 3 个练习文件）
- [ ] `segmentation-gan/README_EN.md` 与中文版结构完全对应
- [ ] 所有 prev/next 导航链接可正确跳转
- [ ] 主 README.md 有语言切换标头
- [ ] 主 README_EN.md 无断链
- [ ] 时间线包含 2017 CycleGAN 和 2018 StyleGAN 条目
- [ ] 所有文件通过 `git status` 验证
