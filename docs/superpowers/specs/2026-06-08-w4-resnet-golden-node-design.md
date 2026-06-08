# W4 · ResNet 金标本节点页（distill.pub 标准 scrollytelling）· 设计

**日期**：2026-06-08
**作者**：通过 brainstorming 共同确定
**状态**：Design Approved，等待写实施计划
**关系**：W1+W2+W3（基础设施，spec `2026-06-07-web-foundation-design.md`）已完成。本 spec 是 W4 — 第一个 distill.pub 标准节点页，作为后续 100+ 节点的视觉模板。

---

## 1. 背景

W3 阶段所有节点页都用通用 `NodePage` 渲染 markdown，质量止步于"GitHub 漂亮一点"。要做到 distill.pub 标准（左侧文字滚动、右侧可视化 sticky、双向交互），必须为**金标本节点**单独建专属 React 组件。

CNN 家族选 **ResNet** 作为第一个金标本——它是大事件节点，已有 SVG 资产，残差直觉天然适合 scrollytelling。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 金标本节点 | **ResNet** |
| 2 | 布局 | **D：双面板都交互**（左 sticky 文字栏含内嵌控件 + 右 sticky 大可视化跟随响应） |
| 3 | 范围 | **MVP（6 个交互单元，3 阶段 × 2 单元）** |
| 4 | 内容融合 | **C：混合路由分流**（`NodePage` 检测金标本路由 → 用专属组件；金标本组件 `import` markdown 文本切片 + JSX 交互） |
| 5 | 技术栈 | **沿用已有**（Framer Motion `useScroll` + d3-scale + 原生 SVG），**不引入** GSAP / Three.js / Canvas / 状态管理库 |
| 6 | 状态管理 | **顶层 React useState**，通过 props 传 |

## 3. 6 个 MVP 交互单元

3 个阶段，每阶段 1 个主可视化 + 1 个内嵌控件：

| # | 阶段 | 对应 Markdown 段落 | 主可视化（右） | 内嵌控件（左） |
|---|------|------|------|------|
| 1.1 | 退化曲线 · 主图 | `## 之前卡在哪` + `## 核心思想`开头 | training error 曲线对比 plain vs ResNet | — |
| 1.2 | 退化曲线 · 控件 | 同上 | — | 层数滑块（8 / 20 / 56 / 110 / 152），曲线实时切换 |
| 2.1 | 残差块拆解 · 主图 | `### 直觉` | 单 Bottleneck 内部（1×1↓ → 3×3 → 1×1↑ + shortcut + ⊕），含 $F(x) + x$ 标注 | — |
| 2.2 | 残差块拆解 · 控件 | 同上 | — | BasicBlock ↔ Bottleneck toggle，参数量数字实时更新 |
| 3.1 | 梯度高速公路 · 主图 | `### 机制` | 6 块串联 + 红虚线（主路梯度衰减） vs 蓝实线（shortcut 梯度恒粗） | — |
| 3.2 | 梯度高速公路 · 控件 | 同上 | — | 堆叠块数滑块（2 / 6 / 20 / 50），主路衰减程度视觉变化 |

## 4. 页面架构

### 4.1 整体布局

```
┌─────────────────────────────────────────────────────────┐
│ Hero 区（不 sticky）                                      │
│  ← 返回 CNN 家族                                         │
│  ResNet (2015)                                          │
│  作者 / 论文 / key_idea                                  │
├─────────────────────────────────────────────────────────┤
│ Sticky 双面板区（占整个滚动主体）                          │
│                                                          │
│ ┌────────────────────┬────────────────────────────────┐│
│ │ 左：文字栏 sticky    │ 右：可视化 sticky                ││
│ │ ↓ 滚动后切换段落      │  根据 active stage 切换内容       ││
│ │                    │                                ││
│ │ 阶段 1 段落 prose +  │  阶段 1: 退化曲线 + 滑块响应       ││
│ │   <DepthSlider/>   │                                ││
│ │ ↓ 滚                │  阶段 2: Bottleneck SVG + 切换    ││
│ │ 阶段 2 段落 prose +  │                                ││
│ │   <BlockTypeToggle/>│  阶段 3: 梯度公路 SVG + 滑块       ││
│ │ ↓ 滚                │                                ││
│ │ 阶段 3 段落 prose +  │                                ││
│ │   <StackDepthSlider/>│                               ││
│ └────────────────────┴────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│ 训练细节（不 sticky · 内联表格 + 内联小图）                  │
├─────────────────────────────────────────────────────────┤
│ 关键代码（不 sticky · 代码块）                              │
├─────────────────────────────────────────────────────────┤
│ 影响 / 后续（不 sticky · 纯文本 + 链接）                    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 滚动驱动机制

- 用 `useScroll({ target: stickyContainerRef })` 监听 sticky 区域滚动进度
- Sticky 容器内 3 个段落，各占容器高度的 1/3
- 当前 active stage 由 `scrollYProgress` 区间决定：
  - [0, 0.33) → stage 1
  - [0.33, 0.66) → stage 2
  - [0.66, 1] → stage 3
- 右面板 `<AnimatePresence mode="wait">` 切换子组件，淡入淡出过渡

### 4.3 阶段切换时的状态保留

每个 widget 状态独立保留——读者在阶段 2 切换 BasicBlock ↔ Bottleneck 后，滚回阶段 1 再回来，仍是上次的选择（state 在 NodePageResNet 顶层 useState 持有）。

## 5. 文件结构

```
web/src/components/node/
├── NodePage.tsx                    ← 修改：在路由分流前检查金标本注册表
└── golden/
    ├── index.ts                    ← 金标本注册表（字符串 → 动态 import）
    └── resnet/
        ├── NodePageResNet.tsx      ← 主组件（页面入口 + 状态）
        ├── NodePageResNet.module.css
        ├── stages/
        │   ├── DegradationStage.tsx     ← 阶段 1（左 prose + DepthSlider + 右 DegradationCurves）
        │   ├── ResidualBlockStage.tsx   ← 阶段 2（左 prose + BlockTypeToggle + 右 BottleneckSVG）
        │   └── GradientHighwayStage.tsx ← 阶段 3（左 prose + StackDepthSlider + 右 GradientHighwaySVG）
        ├── widgets/
        │   ├── DepthSlider.tsx          ← 1.2 网络层数滑块
        │   ├── BlockTypeToggle.tsx      ← 2.2 BasicBlock/Bottleneck toggle
        │   ├── StackDepthSlider.tsx     ← 3.2 堆叠数滑块
        │   ├── DegradationCurves.tsx    ← 1.1 双曲线 SVG
        │   ├── BottleneckSVG.tsx        ← 2.1 残差块 SVG
        │   └── GradientHighwaySVG.tsx   ← 3.1 梯度公路 SVG
        ├── lib/
        │   ├── prose.ts             ← 从 markdown 拆 prose 段（按章节）
        │   └── curves.ts            ← 合成曲线数据公式
        └── NodePageResNet.test.tsx
```

共 13 个新文件。

## 6. 内容融合（决策 C）

### 6.1 NodePage 路由分流

`NodePage.tsx` 改造：

```typescript
import { goldenSamples } from "./golden";

// ...在 useEffect / render 之前：
const goldenKey = `${familyId}/${nodeSlug}`;
const GoldenLoader = goldenSamples[goldenKey];

if (GoldenLoader) {
  // Suspense-style：lazy load 金标本组件
  return (
    <Suspense fallback={<div>Loading golden sample...</div>}>
      <LazyGolden loader={GoldenLoader} />
    </Suspense>
  );
}
// 否则走 W3 markdown 渲染（现有逻辑）
```

### 6.2 金标本注册表

`web/src/components/node/golden/index.ts`：

```typescript
import { lazy } from "react";

export const goldenSamples: Record<string, React.LazyExoticComponent<React.ComponentType>> = {
  "01-cnn/05-resnet": lazy(() => import("./resnet/NodePageResNet")),
  // 未来新增：
  // "05-transformer/01-transformer": lazy(() => import("./transformer/NodePageTransformer")),
};
```

### 6.3 金标本组件读 markdown

`NodePageResNet.tsx`：

```typescript
import resnetMarkdown from "../../../../../../01-cnn/05-resnet.md?raw";
import { extractProse } from "./lib/prose";

const prose = extractProse(resnetMarkdown);
// prose.beforeStuckOn  → "## 之前卡在哪" 章节正文
// prose.coreInsight   → "## 核心思想" 章节正文（不含子标题）
// prose.intuition     → "### 直觉" 子段
// prose.mechanism     → "### 机制" 子段
// prose.training      → "## 训练细节" 章节
// prose.keyCode       → "## 关键代码" 章节（含代码块）
// prose.aftermath     → "## 影响 / 后续" 章节
```

`extractProse` 用简单正则切 H2/H3 章节，返回 plain text 段落数组。每个 stage 组件用 react-markdown 渲染对应段。

允许 5–10% 偏差：金标本组件可以**调整段落顺序**、**省略某些段落**、**新增过渡句**——但**禁止与 markdown 内容矛盾**（事实必须一致）。

## 7. 合成数据

### 7.1 退化曲线

不引用论文真实数据。`lib/curves.ts` 用公式合成：

```typescript
export interface CurvePoint {
  epoch: number;
  loss: number;
}

export function plainCurve(depth: number, epochs = 200): CurvePoint[] {
  // 深度越深越糟：浅模型平滑降到 ~0.05；深模型 (>30 层) 先降后升
  const isDeep = depth > 30;
  const minEpoch = isDeep ? 60 + depth * 0.4 : 30;
  const finalLoss = isDeep ? 0.12 + (depth - 30) * 0.003 : 0.05;
  const minLoss = finalLoss - 0.02;
  // ... 二次/指数衰减 + 反弹
}

export function resnetCurve(depth: number, epochs = 200): CurvePoint[] {
  // 深度越深需要更多 epoch 收敛，但稳定下降到 ~0.04
  // ... 平滑指数衰减
}
```

不必精确——传递"plain 深 30+ 层后训练误差反而升 ; ResNet 152 层仍稳定降"两个直觉就够。

### 7.2 残差块参数量

BasicBlock / Bottleneck 参数量按论文真实公式（C 是通道数）：

- **BasicBlock**（2× conv 3×3 / 64）：`2 × (3 × 3 × 64 × 64) = 73,728 params`
- **Bottleneck**（1×1↓ → 3×3 → 1×1↑ / 64 → 256）：`1×1×256×64 + 3×3×64×64 + 1×1×64×256 = 70,144 params`

显示精确数值，不抽象。

### 7.3 梯度公路衰减

每块的梯度幅度乘以一个固定衰减系数（如 0.85），堆叠数变化时主路梯度按 `0.85^N` 衰减——视觉上线条由粗变细到几乎不见。Shortcut 路径无衰减，恒定粗度。

## 8. 技术依赖

**全部已在 package.json**：

- `framer-motion` ^11 → `useScroll`, `useTransform`, `AnimatePresence`, `motion.div`
- `d3-scale` ^4 → 曲线 x/y scale
- `d3-shape` ^3 → line generator
- `react-markdown` + `remark-gfm` → 渲染 prose 段

**不引入**：

- GSAP / ScrollTrigger
- Three.js / WebGL
- Canvas / pixi.js
- Zustand / Jotai / Redux

## 9. 测试

- `NodePageResNet.test.tsx`（1 个新文件）：
  - mount 不崩
  - 三个 stage 都能渲染
  - widget 触发 state 变化（点击 toggle / 拖滑块），右面板 SVG 重新渲染
  - scroll 切换 stage（用 mock IntersectionObserver / 直接调 setState 测）

scrollytelling 的动画 / sticky 效果 jsdom 不靠谱，**只测 state 流和组件 mount**，视觉部分交给浏览器手动验证。

W3 现有 14 测试不动。新增 1 测试，总 15。

## 10. 路由 & build 影响

### 10.1 路由

- `/families/01-cnn/05-resnet` 走金标本组件
- 其他 99+ 节点继续走 W3 markdown 渲染

### 10.2 Build 大小

金标本组件 lazy import（`React.lazy`），不进主 bundle。访问 ResNet 路由时才下载。

ResNet 金标本组件预计 ~30–50 KB gzip（含 3 个 SVG widget + 几个 prose 段）。不影响主 bundle 的 500 KB gzip 上限。

## 11. 不在本次范围内

- **完整套餐**剩余 5 个交互（1.3 hover loss / 1.4 论文数据点 / 2.3 逐层 hover shape / 2.4 F(x) toggle / 2.5 梯度动画 / 3.3 shortcut on/off / 3.4 数学浮窗）—— 后续 W4.5 单独立项
- **顶配套餐**的梯度反传动画播放、数学浮窗
- **其他金标本节点**（Transformer / CLIP / o1）—— 各自独立 W4-x plan
- **markdown ↔ React 自动同步**（手动维护，允许 5–10% 偏差）
- **暗色模式 / 移动端深度响应式 / a11y 完整审计**
- **静态渲染（SSR/SSG）** —— 仍 SPA
- **论文真实曲线数据** —— 用合成

## 12. 验收标准

1. ✅ 访问 `http://127.0.0.1:5173/families/01-cnn/05-resnet` 加载 `NodePageResNet` 组件（而不是通用 MarkdownRenderer）
2. ✅ 访问其他 7 个 CNN 节点（`02-alexnet` 等）仍走 W3 markdown 渲染
3. ✅ 页面含 3 个 sticky 阶段，滚动时右面板平滑切换可视化
4. ✅ 阶段 1 滑块改变层数 (8/20/56/110/152) → 右图曲线实时切换；plain 在 56+ 层后出现"先降后升"形态
5. ✅ 阶段 2 切换 BasicBlock/Bottleneck → 右图 SVG 结构变化 + 参数量数字 73,728 ↔ 70,144 实时更新
6. ✅ 阶段 3 滑块改变堆叠数 (2/6/20/50) → 右图块数和主路梯度衰减视觉变化（蓝实线恒粗）
7. ✅ Hero / 训练细节 / 关键代码 / 影响后续 4 段非 sticky 内容仍可阅读
8. ✅ `extractProse` 正确从 `01-cnn/05-resnet.md` 切出 7 个章节段
9. ✅ TS 编译零错
10. ✅ `npm run build` 成功；主 bundle gzip 仍 < 500 KB；ResNet chunk lazy load 不进主 bundle
11. ✅ `npm test` 15 测试全过（14 原有 + 1 新）
12. ✅ 浏览器手动验证三个 sticky 阶段切换 + 6 个交互单元都响应

## 13. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **Framer Motion useScroll 在 sticky 上下文行为复杂** | 用 sticky `position: sticky` + Framer `useScroll({ target })` 配合；先做 stage 切换的 PoC 再做精细动画 |
| **左面板内嵌控件不能用 react-markdown 渲染** | prose 段用 ReactMarkdown 渲染，控件以 JSX 兄弟节点形式手工插入到段落之间，不嵌入 markdown |
| **金标本组件文件结构复杂（13 文件）** | 严格按 §5 文件结构分工；主组件只做状态 + 路由，stage/widget 完全独立 |
| **markdown 改了金标本不同步** | 接受 5–10% 偏差；金标本组件代码顶部加注释指明"参照 01-cnn/05-resnet.md，markdown 改动后请同步审视" |
| **lazy import 加载延迟** | Suspense fallback 显示简单 "Loading…"；ResNet 路由首次访问慢 ~200ms 可接受 |
| **scrollytelling 卡在 stage 切换时机** | 用 IntersectionObserver 替代 useScroll 区间方案；多写一些 buffer（每 stage 高度 100vh，切换点在 50vh）|
