# ResNet 残差块 SVG 精品图 · 设计

**日期**：2026-06-07
**作者**：通过 brainstorming 共同确定
**状态**：Design Approved，等待写实施计划
**关系**：补 `2026-06-07-01-cnn-content-design.md` 的 ResNet 节点 SVG TODO。`01-cnn/05-resnet.md` 当前有 2 张 Mermaid + 1 个 `<!-- TODO(SVG) -->` 注释，本 spec 把那张 SVG 落地。

---

## 1. 背景

ResNet 是 CNN 家族的"大事件"节点，按风格指南需要 ≥ 2 张图，至少 1 张 SVG。前一轮 plan 因为 subagent 写 SVG 风险高，把 SVG 标记为 TODO 留到现在。本 spec 把这张 SVG 拍板设计并交付。

这是整个项目**第一张 SVG**——后续其它家族的"特技图"（Transformer Add & Norm、Diffusion UNet skip、CLIP 对齐空间等）会复用本图建立的视觉惯例。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 图焦点 | **D：上下两栏**（上：单块 Bottleneck + 弧形 shortcut；下：6 块串联 + 双梯度路径） |
| 2 | 视觉风格 | **B：强调风（高对比）**——shortcut 蓝色加粗（`#2563eb` 3px），主路用 compute 色 `#fce7f3` |
| 3 | 编号 | 节点正文图 3（图 1/2 是已有的 Mermaid） |

## 3. SVG 规格

### 3.1 文件位置

`01-cnn/assets/05-resnet-residual.svg`

按 `tech-conventions.md §4.5` 命名约定：`<family>/assets/<NN>-<node-name>-<purpose>.svg`。

### 3.2 总尺寸与画布

- `viewBox="0 0 800 600"`
- `preserveAspectRatio="xMidYMid meet"`
- 上栏占用 `y ∈ [0, 360]`（60% 高度）
- 下栏占用 `y ∈ [400, 600]`（约 33%，留 40px 间隔）
- 字体 `font-family="system-ui, -apple-system, sans-serif"`
- 主标签 14px，注释 11px，公式 13px italic

### 3.3 配色（沿用 tech-conventions §1 + 强调色）

| 用途 | fill | stroke | text |
|------|------|--------|------|
| 输入 / 输出节点 | `#fef3c7` | `#d97706` (1.5px) | `#92400e` |
| 计算节点（主路 conv 层） | `#fce7f3` | `#db2777` (1.5px) | `#9d174d` |
| 结果 / 中间状态 | `#ecfdf5` | `#059669` (1.5px) | `#065f46` |
| **shortcut 弧线** | — | `#2563eb` (3px) | `#1e40af` |
| ⊕ 相加节点 | `#fef3c7` | `#d97706` (2px) | `#92400e` |
| 主路细箭头 | — | `#9d174d` (1.5px) | — |
| 梯度反向箭头 · 主路 | — | `#dc2626` (1.5px, dashed) | — |
| 梯度反向箭头 · shortcut | — | `#2563eb` (2.5px) | — |

### 3.4 上栏：单 Bottleneck 块

水平排列，从左到右：

```
Input  →  1×1 Conv ↓C  →  ReLU  →  3×3 Conv  →  ReLU  →  1×1 Conv ↑C  →  ⊕  →  ReLU  →  Output
                                                                          ↑
                                              ┌───────── identity ───────┘  (弧线从 Input 顶部跨到 ⊕ 顶部)
```

- 每个计算节点是圆角矩形 80×40，节点间间距 ~20px
- Input 节点标签：`x`（小字 + 旁注 `[B,256,H,W]`）
- 三个 conv 节点的下方副标签分别是 `64`、`64`、`256`（通道数）
- ⊕ 节点是圆形 r=18，居中标 `⊕` 大字
- Output 节点标签：`y`
- **shortcut 弧形**：用三次贝塞尔曲线从 Input 节点的顶部（约 (60, 200)）弯到 ⊕ 节点上方（约 (640, 200)），最高点在 y=80 附近——形成一道明显高出主路的拱
- 弧线中间偏上贴 `identity` 文字标签（蓝色 italic 13px）
- ⊕ 节点右侧贴 `F(x) + x` 公式（黑色 italic 14px）
- 主路下方加一行集体说明：`F(x) = 1×1↓ → 3×3 → 1×1↑（"瓶颈"结构，节省 ~75% 参数）`（小字 11px）
- 标题（上栏顶部居中）：`Bottleneck Residual Block`（16px）

### 3.5 下栏：梯度高速公路

水平排列 6 个简化残差块（每块用 60×30 的小矩形 + 一道小弧表示），间隔 ~30px。

```
Block 1   Block 2   Block 3   Block 4   Block 5   Block 6
  □__⌒__    □__⌒__    □__⌒__    □__⌒__    □__⌒__    □__⌒__
```

下方两条反向梯度箭头（从右向左）：

- **主路梯度**：红色虚线（`#dc2626` dashed），穿过每个 block 内部（蛇形折线），每段线条**逐段变细**（stroke-width 从 2 渐到 0.5）—— 视觉上传到最左时几乎不见
- **shortcut 梯度**：蓝色实线（`#2563eb` 2.5px），从最右 ⊕ 直接横贯到最左 Input，**线宽不变** —— 直观对比"梯度不衰减"

标注：

- 上方右侧标 `∂L/∂y`（loss 对输出梯度，起点）
- 下方左侧两个箭头终点旁分别标 `∂L/∂x（主路：衰减）`（红字）和 `∂L/∂x（shortcut：不衰减）`（蓝字）
- 下栏标题（顶部居中）：`Gradient Highway · 多块串联时梯度回传路径`（16px）

### 3.6 上下分隔

`y = 380` 画一条 1px 灰色虚线（`#d6d3d1`）作为视觉分隔。

## 4. 节点 markdown 改动

`01-cnn/05-resnet.md` 的改动：

1. 找到 `<!-- TODO(SVG): ... -->` 注释（在 `## 核心思想` 段尾部）
2. 把注释**整行删除**
3. 在原位置插入：

```markdown
![残差块与梯度高速公路](assets/05-resnet-residual.svg)
*图 3：残差块的弧形 shortcut（上）与多块串联的梯度高速公路（下）。*
```

注意：节点正文里现有图 1（ResNet-50 整体 Mermaid）和图 2（Bottleneck 内部 Mermaid）保留不变。新图编号为图 3。

## 5. 验收标准

1. ✅ `01-cnn/assets/05-resnet-residual.svg` 存在，是合法的 SVG（XML 解析成功）
2. ✅ SVG 含 `viewBox="0 0 800 600"`、字体 system-ui sans-serif
3. ✅ 配色严格按 §3.3
4. ✅ 上栏含 7 个主路节点 + 1 个 ⊕ + 1 条弧形 shortcut + `identity` 和 `F(x) + x` 两个文字标签
5. ✅ 下栏含 6 个简化 block + 2 条梯度反向箭头（红虚线 + 蓝实线），且红色每段 stroke-width 递减
6. ✅ 上下栏间有 1px 灰色虚线分隔
7. ✅ `01-cnn/05-resnet.md` 中 `TODO(SVG)` 注释已删除
8. ✅ 节点正文用 `![...](assets/05-resnet-residual.svg)` + `*图 3：...*` caption 引用 SVG
9. ✅ 节点的 13 项原校验仍全过（mermaid 数量仍 ≥ 2，新 SVG 不计入 mermaid 计数）
10. ✅ `TIMELINE.md` 重生成不变（仍 8 nodes，SVG 不影响 frontmatter）
11. ✅ 工作树干净

## 6. 不在本次范围内

- 其它节点的 SVG（仅 ResNet 一张）
- SVG 抽取到共享资产（与 Web CnnTrack.tsx 复用是未来工作）
- 主路梯度衰减的"真实"数学（视觉示意即可）

## 7. 风险与缓解

- **风险 1**：subagent 手写 SVG 容易出现坐标错位、弧形不平滑
  **缓解**：spec 给出精确的坐标和贝塞尔控制点提示；plan 任务 brief 提供完整 `<svg>` 骨架代码，subagent 主要填充内部元素
- **风险 2**：颜色对比度不够，弧形 shortcut 不够突出
  **缓解**：spec §3.3 锁死 shortcut 颜色 `#2563eb` 3px，必跑视觉对比检查
- **风险 3**：SVG 太复杂导致 GitHub 渲染慢
  **缓解**：限制 ~30 个元素以内，单文件 < 10 KB
