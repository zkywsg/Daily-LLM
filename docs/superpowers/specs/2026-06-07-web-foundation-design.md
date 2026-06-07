# Web 顶级视觉化基础设施（W1+W2+W3）· 设计

**日期**：2026-06-07
**作者**：通过 brainstorming 共同确定
**状态**：Design Approved，等待写实施计划
**关系**：本 spec 是 Web 同步项目的第 1 份。后续 W4（CNN 家族金标本节点）+ W5（CNN 其余节点）+ W6+（其他 14 家族）单独立项。

---

## 1. 背景

仓库结构（15 架构家族 + foundations）+ 风格指南 + AlexNet 金标本 + CNN 家族 8 节点 + 1 张 SVG 已全部到位。但 `web/` 子目录仍是旧状态：

- `web/src/data/timeline.ts`（457 行）所有 `modulePath` / `tracks` 指向已归档的 `../tracks/.../`、`../foundations/{deep-learning,math,...}/`——链接全断
- 分类法用旧的 "6 叙事家族"（foundation/vision/language/scale/...）—— 与仓库新主轴的 "15 架构家族" 不对齐
- `timeline.ts` 是手工维护的数据，但仓库已有自动生成的 `TIMELINE.md` + 每节点 frontmatter——双正本

本 spec 把 web 升级为：

- **数据**：单一正本（仓库 markdown frontmatter）
- **分类**：对齐 15 架构家族
- **视觉目标**：定级为 [distill.pub](https://distill.pub) 标准（**最终目标**，本期只建地基）

## 2. 项目分解

Web 顶级视觉化是个多月项目，单 spec 装不下。决定的拆分：

| 子项目 | 目标 | 本 spec 是否覆盖 |
|---|---|---|
| **W1: 数据基础** | `families.json` + TS 类型 + 删除旧数据 | ✓ |
| **W2: 设计系统 + 动效底座** | tokens / 字体 / 色板 / 动效语言 / 依赖 | ✓ |
| **W3: 页面架构** | 路由 + 主页（D 方案）+ 家族页 + 节点页占位 + mini-arches 拆解 | ✓ |
| W4: CNN 家族金标本节点 | 1 个节点做到 distill.pub 标准 | ✗（后续 brainstorm） |
| W5: CNN 其余 7 节点 | 按 W4 模式铺开 | ✗ |
| W6+: 其他 14 家族 | 各家族独立 plan | ✗ |

**本 spec 只覆盖 W1+W2+W3**——目标是搭好"任何人能在地基上往里填顶级视觉内容"的基础设施。

## 3. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 数据源机制 | **A：JSON 中间文件**（`generate_timeline.py` 扩展输出 `families.json`） |
| 2 | 视觉投入定位 | **顶级（V3+V4）**——distill.pub 标准 |
| 3 | 主轴呈现方式 | **D：时间轴为主 + 家族通道切换** |
| 4 | 路由结构 | **C：多页 + 节点页内 scrollytelling** |
| 5 | 现有组件去留 | **B：白板重写 + 保留 mini-arches** |
| 6 | 技术栈 | **推荐栈**：React Router v7 / Framer Motion / CSS Modules + tokens / D3 utilities / lucide-react / Inter+Noto SC+JetBrains Mono |

## 4. 数据模型

### 4.1 `families.json` schema

由 `scripts/generate_timeline.py`（扩展后）从家族目录扫描生成。

```typescript
// web/src/types/family.ts

export type FamilyId =
  | "01-cnn" | "02-rnn-lstm" | "03-word-embedding" | "04-gan"
  | "05-transformer" | "06-bert-family" | "07-gpt-scaling"
  | "08-vit" | "09-multimodal-clip" | "10-diffusion"
  | "11-peft-lora" | "12-rlhf-alignment" | "13-moe-efficient"
  | "14-rag-agent" | "15-reasoning-o1-r1";

export interface NodeData {
  name: string;          // "ResNet"
  year: number;          // 2015
  family: FamilyId;      // "01-cnn"
  order: number;         // 5
  paper: string;         // "Deep Residual Learning..."
  authors: string[];     // ["Kaiming He", ...]
  key_idea: string;      // ≤ 80 chars
  path: string;          // "01-cnn/05-resnet.md" (repo root relative)
  assets: string[];      // ["01-cnn/assets/05-resnet-residual.svg"]
}

export interface FamilyData {
  id: FamilyId;
  label: string;             // "CNN 卷积神经网络"
  blurb: string;             // 一句话定位
  yearRange: [number, number]; // [1998, 2022]
  colorToken: string;        // "--family-01" CSS 变量名
  nodes: NodeData[];         // 按 order 排序
}

export interface FamiliesData {
  generatedAt: string;       // ISO timestamp
  families: FamilyData[];    // 15 个，按 id 排序
}
```

### 4.2 `families.json` 生成规则

`scripts/generate_timeline.py` 扩展：

- 在现有 `collect_nodes` 基础上增加 `collect_families`，扫描每个 `NN-xxx/` 目录
- 从家族 README.md（如果有）提取 H1 行作为 `label`，blockquote 作为 `blurb`
- `yearRange` 从该家族节点的 `year` 字段 min/max 推出
- `colorToken` 按家族 id 映射（`01-cnn` → `--family-01`，固定）
- 输出到 `web/src/data/families.json`
- 同时仍生成 `TIMELINE.md`（双产物）
- 头部加注释 "// AUTO-GENERATED. DO NOT EDIT. Run: python3 scripts/generate_timeline.py"

### 4.3 占位策略（其他 14 家族尚无节点）

未填内容的家族：

- `label` = 家族中文名（从仓库 root README 主表里取）
- `blurb` = "（待补充）"
- `yearRange` = `null` 或基于已知大致年份硬编码
- `nodes` = `[]`
- `colorToken` 仍正常分配

Web 端遇到空家族在 UI 上显示为"灰色待补充"，可点但提示"内容待写"。

## 5. 页面架构

### 5.1 路由

使用 React Router v7。

| URL | 组件 | 说明 |
|---|---|---|
| `/` | `HomePage` | D 方案：时间轴 + 家族切换 |
| `/families/:familyId` | `FamilyPage` | 家族页 |
| `/families/:familyId/:nodeSlug` | `NodePage` | 节点页（W3 占位，W4 重写） |
| `*` | `NotFoundPage` | 404 |

### 5.2 主页（HomePage）· D 方案

**布局：**

```
┌──────────────────────────────────────────────────┐
│ Header (logo + title + 模式 toggle)                │
├──────────────────────────────────────────────────┤
│  [按时间] ⇄ [按家族]    ← 切换 toggle              │
├──────────────────────────────────────────────────┤
│ Main view (随 toggle 切换)                         │
│                                                    │
│  时间模式：横向时间线 1998–2025，节点点按家族着色   │
│  家族模式：15 家族卡片网格，每卡内含子时间线         │
│                                                    │
├──────────────────────────────────────────────────┤
│ Hover/click 弹出 NodeHoverCard：                   │
│   节点名 + 年份 + key_idea + 进入家族 / 进入节点    │
└──────────────────────────────────────────────────┘
```

**交互**：

- 顶部 toggle 切换两种视图——使用 Framer Motion `layoutId` 共享，让同一节点点在两种布局间平滑飞过去
- 时间模式：横向滚动（或缩放），轴用 D3 `scaleLinear` 把年份映射到 x 坐标
- 家族模式：15 张卡片，每张内含本家族 mini-arch 链 + 节点 hover 详情
- 当前可见的"按时间"是默认模式

**组件**：

- `HomePage.tsx`: 根组件 + 模式状态
- `TimeAxisView.tsx`: 时间模式视图
- `FamilyGridView.tsx`: 家族模式视图
- `NodeHoverCard.tsx`: 悬浮卡片

### 5.3 家族页（FamilyPage）

`/families/01-cnn`：

```
┌──────────────────────────────────────────────────┐
│ ← 返回主页                                         │
│ 01-cnn · CNN 卷积神经网络                          │
│ > 把图像从手工特征解放出来，让模型自己学层级表征     │
├──────────────────────────────────────────────────┤
│ 概念本身（render 家族 README "概念本身" 段）        │
├──────────────────────────────────────────────────┤
│ 子时间线（横向）                                   │
│ [LeNet][AlexNet][VGG][Inception][ResNet]...      │
│  ↑每个节点显示 mini-arch + 关键贡献                │
├──────────────────────────────────────────────────┤
│ 依赖与延伸                                         │
│ 前置 foundations / 通向其他家族                    │
└──────────────────────────────────────────────────┘
```

实现：

- 路由参数 `familyId` 查 `families.json`，找不到 → 404
- 家族 README markdown 用 react-markdown 渲染（Mermaid 块跳过，W4 再处理）
- 子时间线含 mini-arch 缩略图（来自 `mini-arches/` 目录）+ 节点链接

### 5.4 节点页占位（NodePage）

`/families/01-cnn/05-resnet`：

W3 阶段只做"高质量 markdown 渲染"。Mermaid 用客户端 mermaid.js 渲染（动态 import 避免主 bundle 变大）。SVG 用 `<img>` 加载。

```
┌──────────────────────────────────────────────────┐
│ ← 返回家族                                         │
│ ResNet (2015)                                     │
│ 作者: Kaiming He et al. · 论文: ...                │
├──────────────────────────────────────────────────┤
│ 渲染节点 markdown 全文                             │
│  - frontmatter → 顶部 metadata                     │
│  - ## 标题 → 锚点导航                              │
│  - ```mermaid 块 → mermaid.js 渲染                │
│  - ![](assets/*.svg) → 直接 <img>                  │
│  - $$..$$ → KaTeX (rehype-katex 已装)              │
│  - ```python → highlight.js (已装)                 │
└──────────────────────────────────────────────────┘
```

W4 阶段会替换为定制 scrollytelling 组件。

## 6. 设计系统

### 6.1 Design tokens（`web/src/styles/tokens.css`）

```css
:root {
  /* —— 颜色 · 中性 —— */
  --bg-canvas: #fafafa;
  --bg-surface: #ffffff;
  --bg-subtle: #f4f4f5;
  --ink-primary: #18181b;
  --ink-secondary: #52525b;
  --ink-muted: #a1a1aa;
  --border: #e4e4e7;

  /* —— 颜色 · 强调（沿用 tech-conventions §1） —— */
  --accent-input: #fef3c7;
  --accent-input-line: #d97706;
  --accent-input-ink: #92400e;
  --accent-compute: #fce7f3;
  --accent-compute-line: #db2777;
  --accent-compute-ink: #9d174d;
  --accent-output: #ecfdf5;
  --accent-output-line: #059669;
  --accent-output-ink: #065f46;
  --accent-link: #2563eb;
  --accent-link-ink: #1e40af;
  --accent-warn: #ea580c;

  /* —— 15 家族色板 —— */
  /* 暖色起（视觉系），冷色收（推理系），按时间渐变 */
  --family-01: #db2777; /* CNN 玫红 */
  --family-02: #e11d48; /* RNN/LSTM 红 */
  --family-03: #f97316; /* Word Embedding 橙 */
  --family-04: #f59e0b; /* GAN 琥珀 */
  --family-05: #eab308; /* Transformer 金 */
  --family-06: #84cc16; /* BERT 黄绿 */
  --family-07: #22c55e; /* GPT/Scaling 绿 */
  --family-08: #10b981; /* ViT 翠 */
  --family-09: #14b8a6; /* CLIP 青 */
  --family-10: #06b6d4; /* Diffusion 蓝绿 */
  --family-11: #0ea5e9; /* PEFT 天蓝 */
  --family-12: #3b82f6; /* RLHF 蓝 */
  --family-13: #6366f1; /* MoE 靛 */
  --family-14: #8b5cf6; /* RAG/Agent 紫 */
  --family-15: #a855f7; /* o1/R1 推理 紫红 */

  /* —— 字体 —— */
  --font-sans: "Inter", "Noto Sans SC", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", "Courier New", monospace;
  --font-serif: "Source Serif Pro", Georgia, serif; /* 节点正文长阅读用 */

  /* —— 字号 · 模数 1.25 —— */
  --fs-xs: 0.75rem;
  --fs-sm: 0.875rem;
  --fs-base: 1rem;
  --fs-md: 1.125rem;
  --fs-lg: 1.25rem;
  --fs-xl: 1.5rem;
  --fs-2xl: 1.875rem;
  --fs-3xl: 2.25rem;
  --fs-4xl: 3rem;
  --fs-5xl: 4rem;

  /* —— 间距 · 基于 4px —— */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-12: 3rem;
  --space-16: 4rem;
  --space-24: 6rem;
  --space-32: 8rem;

  /* —— 圆角 —— */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-full: 9999px;

  /* —— 阴影 —— */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  --shadow-lg: 0 12px 32px rgba(0,0,0,0.12);

  /* —— 动效曲线 —— */
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in-out: cubic-bezier(0.65, 0, 0.35, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --dur-fast: 150ms;
  --dur-base: 250ms;
  --dur-slow: 400ms;
}

[data-theme="dark"] {
  --bg-canvas: #09090b;
  --bg-surface: #18181b;
  --bg-subtle: #27272a;
  --ink-primary: #fafafa;
  --ink-secondary: #d4d4d8;
  --ink-muted: #71717a;
  --border: #3f3f46;
  /* 家族色板在 dark 下保持不变（已经足够鲜艳）*/
}
```

### 6.2 字体托管

通过 Fontsource self-host（避免 CDN 单点故障，且 GDPR 友好）：

```json
{
  "@fontsource/inter": "^5.x",
  "@fontsource/noto-sans-sc": "^5.x",
  "@fontsource/jetbrains-mono": "^5.x",
  "@fontsource/source-serif-pro": "^5.x"
}
```

`web/src/styles/fonts.css` 集中 import。

### 6.3 动效语言（`web/src/lib/motion.ts`）

```typescript
export const motion = {
  duration: { fast: 0.15, base: 0.25, slow: 0.4 },
  ease: {
    out: [0.16, 1, 0.3, 1] as const,
    inOut: [0.65, 0, 0.35, 1] as const,
    spring: [0.34, 1.56, 0.64, 1] as const,
  },
  // Framer Motion 常用 variant 预设
  fadeUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  },
  stagger: {
    container: { animate: { transition: { staggerChildren: 0.05 } } },
  },
};
```

### 6.4 色板工具（`web/src/lib/colors.ts`）

```typescript
export function familyColorVar(id: FamilyId): string {
  const num = id.split("-")[0]; // "01"
  return `var(--family-${num})`;
}
```

## 7. 技术栈

### 7.1 新增依赖

```json
{
  "dependencies": {
    "react-router": "^7.0.0",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.400.0",
    "d3-scale": "^4.0.2",
    "d3-shape": "^3.2.0",
    "d3-axis": "^3.0.0",
    "@fontsource/inter": "^5.0.0",
    "@fontsource/noto-sans-sc": "^5.0.0",
    "@fontsource/jetbrains-mono": "^5.0.0",
    "@fontsource/source-serif-pro": "^5.0.0",
    "mermaid": "^11.0.0"
  },
  "devDependencies": {
    "@types/d3-scale": "^4.0.0",
    "@types/d3-shape": "^3.1.0",
    "@types/d3-axis": "^3.0.0"
  }
}
```

### 7.2 明确不引入

- **Tailwind**——sticking CSS Modules + tokens
- **GSAP**——Framer Motion 内置 `useScroll` 够用
- **Three.js / WebGL**——CNN 用 2D SVG 足够
- **完整 D3**——只用 utility 函数

## 8. 文件结构

### 8.1 新建

```
web/src/
├── data/
│   └── families.json                  ← W1 生成（脚本写入）
├── types/
│   └── family.ts                      ← W1
├── styles/
│   ├── tokens.css                     ← W2
│   ├── fonts.css                      ← W2
│   └── global.css                     ← W2（重置 + 基础排版）
├── lib/
│   ├── motion.ts                      ← W2
│   └── colors.ts                      ← W2
├── components/
│   ├── mini-arches/                   ← W3（拆解自 CnnTrack.tsx）
│   │   ├── MiniLeNet.tsx
│   │   ├── MiniAlexNet.tsx
│   │   ├── MiniVGG.tsx
│   │   ├── MiniGoogLeNet.tsx
│   │   ├── MiniResNet.tsx
│   │   ├── MiniDenseNet.tsx
│   │   ├── MiniEfficientNet.tsx
│   │   ├── MiniConvNeXt.tsx
│   │   └── MiniSENet.tsx              （CnnTrack 已有，保留）
│   ├── home/
│   │   ├── HomePage.tsx
│   │   ├── TimeAxisView.tsx
│   │   ├── FamilyGridView.tsx
│   │   └── NodeHoverCard.tsx
│   ├── family/
│   │   └── FamilyPage.tsx
│   ├── node/
│   │   ├── NodePage.tsx               ← W3 占位（W4 重写）
│   │   ├── MarkdownRenderer.tsx       ← react-markdown + plugins 封装
│   │   └── MermaidBlock.tsx           ← 动态 import mermaid.js
│   └── ui/
│       ├── Layout.tsx                 ← 全局头部 + 容器
│       └── NotFoundPage.tsx
└── App.tsx                            ← 改造为路由根
```

### 8.2 删除

```
web/src/
├── components/
│   ├── App.tsx (旧)                   ← 替换为新 App.tsx
│   ├── TimelineAxis.tsx
│   ├── TimelineAxis.test.tsx
│   ├── TimelineContent.tsx
│   ├── TimelineWorkList.tsx
│   ├── RelatedModules.tsx
│   ├── ArchitectureMap.tsx
│   ├── TimelineIllustration.tsx
│   ├── PrehistoryDrawer.tsx
│   ├── TrackView.tsx
│   └── tracks/CnnTrack.tsx            ← mini-arches 已拆出，整文件删
├── data/
│   ├── timeline.ts
│   ├── timeline.test.ts
│   └── phaseFamily.ts
└── App.test.tsx                       ← 重写为新路由测试
```

### 8.3 保留

```
web/
├── index.html
├── package.json (更新依赖)
├── package-lock.json
├── tsconfig*.json
├── vite.config.ts
└── src/
    ├── main.tsx
    └── vite-env.d.ts
```

## 9. 测试策略

### 9.1 W1

- 扩展 `scripts/test_generate_timeline.py` 加 1 个测试：fixture 仓库扫描后 `families.json` 含正确的 family 数量 + node 列表 + 类型正确
- families.json 生成后跑 TypeScript 编译，确保和 `family.ts` 类型对上

### 9.2 W2

- tokens.css 没有专门测试（CSS 变量人工 spot-check）
- `lib/motion.ts` 和 `lib/colors.ts` 各加 1–2 个单测

### 9.3 W3

- `App.test.tsx` 重写：测试 3 个主要路由能 mount 不报错
- `HomePage.test.tsx`：测试模式 toggle 切换；hover 节点弹卡片
- `FamilyPage.test.tsx`：测试 `01-cnn` 路由能渲染家族 + 8 节点
- `NodePage.test.tsx`：测试 markdown 渲染 + Mermaid 占位
- `mini-arches/`：8 个组件 smoke render 测试

约 12 个测试，覆盖率不追求 100%，但关键 happy path 必须过。

## 10. 不在本次范围内

- W4 金标本节点（节点页 scrollytelling 定制可视化）
- W5+ 其他节点 / 家族的内容填充
- SSR / SSG 部署优化（本期 SPA，部署 GitHub Pages 也行）
- Dark mode（CSS variables 已留 hook，组件按需启用）
- 国际化（i18n）——目前只中文
- 移动端响应式细节（基本布局响应即可，移动端深度优化未来再说）
- ARIA 完整 a11y 覆盖（关键交互保证可用，screen reader 优化未来）

## 11. 验收标准

1. ✅ `scripts/generate_timeline.py` 扩展后产出 `web/src/data/families.json`，包含 15 个家族 + CNN 8 节点真实数据，其余 14 家族占位
2. ✅ `web/src/types/family.ts` 与 JSON schema 一致，TS 编译通过
3. ✅ 旧 `timeline.ts` / `phaseFamily.ts` / 旧组件全部删除
4. ✅ `web/src/styles/tokens.css` 含 6.1 §的所有 tokens（含 15 家族色板）
5. ✅ 自托管字体 import 链路通（Inter + Noto Sans SC + JetBrains Mono + Source Serif Pro）
6. ✅ 8 个 mini-arch 组件从旧 `CnnTrack.tsx` 拆出可独立 import
7. ✅ 路由结构就位：`/` / `/families/:id` / `/families/:id/:slug` / `*`
8. ✅ 主页 D 方案可用：时间模式 / 家族模式可切换，节点点可 hover 弹卡片
9. ✅ 家族页 `/families/01-cnn` 能渲染家族 README + 8 节点子时间线 + mini-arch 缩略图
10. ✅ 节点页 `/families/01-cnn/05-resnet` 能渲染节点 markdown（含 Mermaid + SVG + 公式 + 代码）
11. ✅ `npm run build` 成功；产物 < 500 KB gzip
12. ✅ `npm test` 12 个测试全过
13. ✅ `npm run dev` 在 `http://127.0.0.1:5173/` 启动，浏览器手动验证 3 个核心路由

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **Mermaid client-side 渲染过大** | 动态 import，按需加载（节点页才下载） |
| **15 家族色板鲜艳过载** | 默认低饱和应用（卡片描边 / 标签底色），仅强调时用满饱和 |
| **CnnTrack.tsx 拆解出现 SVG 错位** | 拆解后逐个 visual smoke test，先重定位再批量改 |
| **react-router v7 API 变化** | 锁版本号 `^7.0.0`，遵循 dataLoader 新规范 |
| **Framer Motion bundle 偏大** | 用 `motion/react` 子包，按需 tree-shake |
| **families.json schema 后续要扩** | 把 schema 字段标 optional 优先，新增不破坏旧消费者 |
