# 世界模型 / 视频生成家族金标本交互页 Implementation Plan

**Goal:** 为第 16 个家族(`16-world-models`)的 6 个节点(World Models/Video Diffusion Models/DreamerV3/Sora/Genie/GameNGen)各自补一个交互金标本页,复用现有 `web/src/components/node/golden/{slug}/` 结构约定。

**Architecture:** 每节点 `lib/data.ts`(确定性 demo 数据/函数)+ `lib/prose.ts`(mixtral 式双层提取模板)+ 3 个 `stages/*.tsx`(对应机制一/二/三,每个 stage 用 1 个自包含 widget,内部管理自己的交互状态)+ `NodePage{Name}.tsx`(hero+3 stage+footer)+ `NodePage{Name}.module.css`,注册进 `web/src/components/node/golden/index.ts`。

**Tech Stack:** React + TypeScript,内联 SVG/HTML(参照 `golden/mixtral`、`golden/gcn` 的既有写法),CSS Modules,Vitest。

---

## 关键背景(所有任务共用)

**本家族的 markdown 结构与 mixtral 相同**(与更晚的 17/18 家族的扁平结构不同):`## 核心思想:xxx` 下嵌套 `### 直觉` 和 `### N 个必须同时跨过的坎` 两个 H3,`## 机制一/二/三` 是顶层 H2。`lib/prose.ts` 直接复用 `web/src/components/node/golden/mixtral/lib/prose.ts` 的双层提取逻辑:

```typescript
export interface ProseSections {
  previousWork: string;
  intuition: string;
  mechanism1: string;
  mechanism2: string;
  mechanism3: string;
  synergy: string;
  keyCode: string;
  performance: string;
  aftermath: string;
}

const H3_KEYS: Array<{ test: RegExp; key: keyof ProseSections }> = [
  { test: /^直觉/, key: "intuition" },
];

const H2_KEYS: Array<{ test: RegExp; key: keyof ProseSections | "_coreInsight" }> = [
  { test: /^前作进展/, key: "previousWork" },
  { test: /^核心思想/, key: "_coreInsight" },
  { test: /^机制一/, key: "mechanism1" },
  { test: /^机制二/, key: "mechanism2" },
  { test: /^机制三/, key: "mechanism3" },
  { test: /^三件套协同/, key: "synergy" },
  { test: /^关键代码/, key: "keyCode" },
  { test: /^性能数据/, key: "performance" },
  { test: /^影响/, key: "aftermath" },
];

export function extractProse(markdown: string): ProseSections {
  const body = markdown
    .replace(/^---[\s\S]*?---\n?/, "")
    .replace(/```mermaid\n[\s\S]*?```\n(\*图 ?\d[^\n]*\*\n?)?/g, "");

  const sections: ProseSections = {
    previousWork: "", intuition: "", mechanism1: "", mechanism2: "",
    mechanism3: "", synergy: "", keyCode: "", performance: "", aftermath: "",
  };

  let currentKey: keyof ProseSections | null = null;
  let inCoreInsight = false;
  let buffer: string[] = [];

  const flush = () => {
    if (currentKey) sections[currentKey] = buffer.join("\n").trim();
    buffer = [];
  };

  for (const line of body.split("\n")) {
    const h2 = /^## +(.+?)\s*$/.exec(line);
    const h3 = /^### +(.+?)\s*$/.exec(line);

    if (h2) {
      flush();
      const m = H2_KEYS.find((x) => x.test.test(h2[1].trim()));
      if (m && m.key === "_coreInsight") {
        currentKey = null;
        inCoreInsight = true;
      } else if (m) {
        currentKey = m.key as keyof ProseSections;
        inCoreInsight = false;
      } else {
        currentKey = null;
        inCoreInsight = false;
      }
      continue;
    }

    if (h3 && inCoreInsight) {
      flush();
      const m = H3_KEYS.find((x) => x.test.test(h3[1].trim()));
      currentKey = m ? m.key : null;
      continue;
    }

    if (currentKey) buffer.push(line);
  }
  flush();

  return sections;
}
```

每个节点的 `lib/prose.ts` 只需改 `XXX_SOURCE_PATH` 常量,`extractProse` 逻辑原样复制。**写每个节点前先跑 `grep -n '^## \|^### ' 16-world-models/{文件}.md` 确认标题措辞与上面正则匹配**(尤其"三个/两个必须同时跨过的坎"这个 H3 不参与提取,跳过即可,不影响 intuition 提取)。

**路由注册 key 格式**:`"16-world-models/{NN-slug}"`(如 `"16-world-models/01-world-models"`)。

**标题渐变色**:`.title` 用 `linear-gradient(90deg, var(--family-16) 0%, var(--accent-link) 100%)`(family-16 = `#d946ef` 洋红)。

**通用 `NodePage{Name}.module.css` 模板**(6 个节点完全复制,只改渐变起始色变量名固定为 `--family-16`):

```css
.container {
  padding: 0;
  background: var(--bg-canvas);
}

.hero {
  max-width: 1000px;
  margin: 0 auto;
  padding: var(--space-8) var(--space-4);
  text-align: center;
}

@media (min-width: 768px) {
  .hero {
    padding: var(--space-12) var(--space-8);
  }
}

.back {
  display: inline-block;
  margin-bottom: var(--space-6);
  font-size: var(--fs-sm);
  color: var(--ink-secondary);
}

.title {
  font-size: clamp(2rem, 5vw + 1rem, 4rem);
  margin: var(--space-4) 0;
  background: linear-gradient(90deg, var(--family-16) 0%, var(--accent-link) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.metaLine {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
  margin: var(--space-1) 0;
}

.keyIdea {
  font-size: var(--fs-lg);
  color: var(--ink-secondary);
  font-style: italic;
  max-width: 700px;
  margin: var(--space-6) auto 0;
  line-height: 1.5;
}

.stage {
  max-width: 1300px;
  margin: 0 auto;
  padding: var(--space-12) var(--space-4);
  min-height: 100vh;
}

@media (min-width: 768px) {
  .stage {
    padding: var(--space-24) var(--space-8);
  }
}

.stageAlt {
  background: var(--bg-surface);
}

.footer {
  max-width: 800px;
  margin: 0 auto;
  padding: var(--space-12) var(--space-4);
  font-family: var(--font-serif);
  font-size: var(--fs-md);
  line-height: 1.7;
}

@media (min-width: 768px) {
  .footer {
    padding: var(--space-16) var(--space-8);
  }
}

.footerSection {
  margin: var(--space-12) 0;
}

.footerSection h2 {
  font-family: var(--font-sans);
  font-size: var(--fs-2xl);
  margin-bottom: var(--space-4);
}
```

**通用 `stages/Stage.module.css` 模板**(6 个节点完全复制):

```css
.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-6);
  align-items: start;
}

.stickyPanel {
  position: static;
  width: 100%;
}

@media (min-width: 1024px) {
  .grid {
    grid-template-columns: 1fr 1fr;
    gap: var(--space-8);
  }

  .stickyPanel {
    position: sticky;
    top: var(--space-8);
  }
}

.caption {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
  margin-top: var(--space-2);
  line-height: 1.5;
}
```

**Stage 组件通用骨架**(每个 stage 内部结构一致,只是 widget 换掉):

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { XXX_SOURCE_PATH } from "../lib/prose";
import { SomeWidget } from "../widgets/SomeWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanismNProse: string;
  // Stage1 额外接 intuitionProse,Stage3 额外接 synergyProse
}

export function SomeStage({ mechanismNProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制N:{"{标题}"}
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        {"{一句话引导语}"}
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanismNProse} sourcePath={XXX_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <SomeWidget />
        </div>
      </div>
    </div>
  );
}
```

**已知坑(前几轮沉淀)**:
1. SVG 里的条形图/曲线高度必须 `Math.min(..., cap)` 钳位,禁止无上限缩放
2. 所有 toggle/选择按钮加 `aria-pressed={condition}`
3. `**` 靠近标点时确保另一侧是空白(本任务只涉及 TSX 内联字符串和 CSS,不改动 markdown 正本,原则上不涉及此坑,但组件里若拼接展示 markdown 片段仍需留意)
4. `NodePage.tsx` 路由解析用 `${familyId}/${nodeSlug}`,`index.ts` 注册 key 必须精确匹配

---

## Task 1: World Models — lib + Stage1(V)+ Stage2(M)

**Files:**
- Create: `web/src/components/node/golden/world-models/lib/data.ts`
- Create: `web/src/components/node/golden/world-models/lib/prose.ts`
- Create: `web/src/components/node/golden/world-models/stages/Stage.module.css`
- Create: `web/src/components/node/golden/world-models/widgets/VaeCompressWidget.tsx`
- Create: `web/src/components/node/golden/world-models/stages/VisionStage.tsx`
- Create: `web/src/components/node/golden/world-models/widgets/MixtureDensityWidget.tsx`
- Create: `web/src/components/node/golden/world-models/stages/MemoryStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## \|^### ' 16-world-models/01-world-models.md
```

Expected:含 `## 前作进展`、`## 核心思想:...`(下有 `### 直觉` 和 `### 三个必须同时跨过的坎`)、`## 机制一:V(Vision)—— VAE 视觉压缩`、`## 机制二:M(Memory)—— MDN-RNN 时序动态预测`、`## 机制三:C(Controller)—— 极小线性控制器,完全在梦境里训练`、`## 三件套协同`、`## 关键代码`、`## 性能数据`、`## 影响 / 后续`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// World Models demo 数据:8x8 toy "帧"(灰度网格)+ VAE 压缩/重建 +
// MDN(混合高斯)预测下一潜状态的确定性函数。不真跑训练。

export const GRID_SIZE = 8;

/** 一个固定的 toy 帧:8x8 灰度值(0-1),模拟一个简单场景 */
export const TOY_FRAME: number[] = Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, i) => {
  const x = i % GRID_SIZE;
  const y = Math.floor(i / GRID_SIZE);
  const cx = 3.5, cy = 3.5;
  const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
  return Math.max(0, 1 - d / 4);
});

/** 用简单哈希把 64 维帧压缩成 latentDim 维潜向量(确定性,模拟 VAE 编码) */
export function encodeVAE(frame: number[], latentDim: number): number[] {
  const z: number[] = [];
  for (let d = 0; d < latentDim; d++) {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) {
      const w = Math.sin((i + 1) * (d + 1) * 0.37) * 0.5;
      sum += frame[i] * w;
    }
    z.push(sum / frame.length);
  }
  return z;
}

/** 从潜向量近似重建帧:latentDim 越小,重建越模糊(信息损失越大) */
export function decodeVAE(z: number[], latentDim: number): number[] {
  const recon: number[] = [];
  for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
    let sum = 0;
    for (let d = 0; d < latentDim; d++) {
      const w = Math.sin((i + 1) * (d + 1) * 0.37) * 0.5;
      sum += z[d] * w;
    }
    // latentDim 越大,重建越接近原图;用一个模糊系数模拟维度不足的信息损失
    const fidelity = Math.min(1, latentDim / 16);
    recon.push(Math.max(0, Math.min(1, sum * fidelity + 0.5 * (1 - fidelity))));
  }
  return recon;
}

export interface MixtureComponent {
  mean: number;
  std: number;
  weight: number;
}

/** 给定当前潜状态 z 的第 0 维,用 K 个高斯分量模拟 MDN-RNN 预测的下一状态分布 */
export function predictMixture(z0: number, k: number, seed = 0): MixtureComponent[] {
  const comps: MixtureComponent[] = [];
  for (let i = 0; i < k; i++) {
    let h = (seed + i * 977 + Math.round(z0 * 1000) * 31) >>> 0;
    h = (h * 2654435761) >>> 0;
    const mean = z0 + (((h % 1000) / 1000) * 2 - 1) * 0.8;
    const std = 0.1 + ((h >>> 8) % 100) / 1000;
    comps.push({ mean, std, weight: 0 });
  }
  // softmax 权重,确定性
  const rawWeights = comps.map((_, i) => Math.exp(Math.sin((seed + i + 1) * 1.3)));
  const sum = rawWeights.reduce((a, b) => a + b, 0);
  comps.forEach((c, i) => (c.weight = rawWeights[i] / sum));
  return comps;
}

/** 从混合分布采样出下一个 z0(确定性:取加权期望而非随机采样,便于演示复现) */
export function sampleMixtureMean(comps: MixtureComponent[]): number {
  return comps.reduce((s, c) => s + c.mean * c.weight, 0);
}

/** 梦境 rollout:给定初始 z0,自回归调用 predictMixture + sampleMixtureMean 前进 N 步,
 * 全程不接触真实帧/环境 —— 这就是 World Models 论文"完全在梦境里训练 C"的核心机制。 */
export function dreamRollout(z0Start: number, steps: number): number[] {
  const traj = [z0Start];
  let z0 = z0Start;
  for (let t = 0; t < steps; t++) {
    const comps = predictMixture(z0, 5, t);
    z0 = sampleMixtureMean(comps);
    traj.push(z0);
  }
  return traj;
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用"关键背景"里的通用双层模板,加:

```typescript
export const WORLD_MODELS_SOURCE_PATH = "16-world-models/01-world-models.md";
```

- [ ] **Step 4: 写 `stages/Stage.module.css`**

原样复制"关键背景"模板。

- [ ] **Step 5: 写 `widgets/VaeCompressWidget.tsx`**

```tsx
import { useState } from "react";
import { GRID_SIZE, TOY_FRAME, encodeVAE, decodeVAE } from "../lib/data";

// 8x8 toy 帧 → 可调维度的潜向量 → 重建。潜维度越小重建越模糊,
// 直观展示 VAE 压缩的信息损失权衡。

function GridSvg({ values, size = 160 }: { values: number[]; size?: number }) {
  const cell = size / GRID_SIZE;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {values.map((v, i) => {
        const x = (i % GRID_SIZE) * cell;
        const y = Math.floor(i / GRID_SIZE) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${g},${g})`} stroke="var(--bg-canvas)" strokeWidth={0.5} />;
      })}
    </svg>
  );
}

export function VaeCompressWidget() {
  const [latentDim, setLatentDim] = useState(8);
  const z = encodeVAE(TOY_FRAME, latentDim);
  const recon = decodeVAE(z, latentDim);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        潜向量维度 = {latentDim}
        <input type="range" min={1} max={16} value={latentDim} onChange={(e) => setLatentDim(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", gap: "var(--space-6)", alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>原始帧(8×8)</div>
          <GridSvg values={TOY_FRAME} />
        </div>
        <div style={{ fontSize: "var(--fs-2xl)", color: "var(--ink-muted)" }}>→</div>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>z(潜向量,{latentDim} 维)</div>
          <div style={{ display: "flex", gap: 2, flexWrap: "wrap", width: 160 }}>
            {z.map((v, i) => (
              <div key={i} title={v.toFixed(2)} style={{ width: 16, height: 16, background: `hsl(${v > 0 ? 200 : 0}, 70%, ${60 - Math.min(Math.abs(v) * 40, 30)}%)`, borderRadius: 2 }} />
            ))}
          </div>
        </div>
        <div style={{ fontSize: "var(--fs-2xl)", color: "var(--ink-muted)" }}>→</div>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>重建帧</div>
          <GridSvg values={recon} />
        </div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        维度越小,重建越模糊(信息损失越大);维度越大,重建越接近原图,但 C 后续要处理的状态空间也越大。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/VisionStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORLD_MODELS_SOURCE_PATH } from "../lib/prose";
import { VaeCompressWidget } from "../widgets/VaeCompressWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function VisionStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:V(Vision)—— VAE 视觉压缩
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把高维原始帧压缩成低维潜向量 z,后续 M 和 C 全部在这个压缩后的潜空间里工作,而不是直接处理像素。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <VaeCompressWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/MixtureDensityWidget.tsx`**

```tsx
import { useState } from "react";
import { predictMixture, sampleMixtureMean } from "../lib/data";

const W = 680;
const H = 280;

// 给定当前 z0,用 K 个高斯分量可视化 MDN-RNN 预测的下一状态分布——
// 混合分量数越多,能表达的"下一步可能走向"越多样(多峰)。

export function MixtureDensityWidget() {
  const [k, setK] = useState(3);
  const z0 = 0.2;
  const comps = predictMixture(z0, k);
  const mean = sampleMixtureMean(comps);

  const xMin = -1.5, xMax = 1.5;
  const toX = (v: number) => ((v - xMin) / (xMax - xMin)) * (W - 60) + 30;
  const points = Array.from({ length: 200 }, (_, i) => xMin + (i / 199) * (xMax - xMin));
  const density = points.map((x) =>
    comps.reduce((s, c) => s + c.weight * Math.exp(-((x - c.mean) ** 2) / (2 * c.std ** 2)) / (c.std * Math.sqrt(2 * Math.PI)), 0)
  );
  const maxD = Math.max(...density, 0.1);
  const toY = (d: number) => H - 40 - Math.min((d / maxD) * (H - 80), H - 80);

  const path = points.map((x, i) => `${i === 0 ? "M" : "L"} ${toX(x)} ${toY(density[i])}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        混合分量数 K = {k}
        <input type="range" min={1} max={6} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`K=${k} 个高斯分量混合的下一状态预测分布`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          下一状态 z' 的预测分布(K={k} 个高斯分量混合)
        </text>
        <line x1={30} y1={H - 40} x2={W - 30} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
        {comps.map((c, i) => (
          <circle key={i} cx={toX(c.mean)} cy={H - 40} r={3 + c.weight * 10} fill="#d946ef" opacity={0.5} />
        ))}
        <line x1={toX(mean)} y1={30} x2={toX(mean)} y2={H - 40} stroke="var(--ink-muted)" strokeDasharray="3 3" />
        <text x={toX(mean)} y={26} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">期望值</text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        K 越大,分布能表达的"下一步走向"越多样(多峰);K=1 退化成单一高斯,只能预测一种确定性走向。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/MemoryStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORLD_MODELS_SOURCE_PATH } from "../lib/prose";
import { MixtureDensityWidget } from "../widgets/MixtureDensityWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function MemoryStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:M(Memory)—— MDN-RNN 时序动态预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        M 不是预测确定的下一状态,而是预测一个混合高斯分布——世界是有随机性的,同一个当前状态可能走向多种不同的未来。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WORLD_MODELS_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <MixtureDensityWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

```bash
cd web && npx tsc --noEmit
```

Expected: 无错误(这些文件尚未被 index.ts 引用,不应报错)。

- [ ] **Step 10: 提交**

```bash
git add web/src/components/node/golden/world-models/lib web/src/components/node/golden/world-models/stages web/src/components/node/golden/world-models/widgets
git commit -m "feat: World Models 金标本 lib + Stage1/2"
```

---

## Task 2: World Models — Stage3(C)+ NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/world-models/widgets/DreamRolloutWidget.tsx`
- Create: `web/src/components/node/golden/world-models/stages/ControllerStage.tsx`
- Create: `web/src/components/node/golden/world-models/NodePageWorldModels.tsx`
- Create: `web/src/components/node/golden/world-models/NodePageWorldModels.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/DreamRolloutWidget.tsx`**

```tsx
import { useState } from "react";
import { dreamRollout } from "../lib/data";

const W = 680;
const H = 260;

// 完全在 M 自回归生成的"梦境"轨迹上跑 rollout:z0 → M → z1 → M → z2 → …,
// 全程不接触真实帧/环境。点"梦境前进一步"每次追加一步。

export function DreamRolloutWidget() {
  const [steps, setSteps] = useState(0);
  const traj = dreamRollout(0.2, steps);

  const xMin = 0, xMax = 10;
  const yMin = -1.5, yMax = 1.5;
  const toX = (t: number) => 40 + (t / xMax) * (W - 80);
  const toY = (v: number) => H - 40 - ((v - yMin) / (yMax - yMin)) * (H - 80);

  const path = traj.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`梦境 rollout,已进行 ${steps} 步`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          纯梦境 rollout(z0 → M → z1 → M → z2 → …),已进行 {steps} 步
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <line x1={40} y1={30} x2={40} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} strokeDasharray="4 2" />
        {traj.map((v, i) => (
          <circle key={i} cx={toX(i)} cy={toY(v)} r={4} fill="#d946ef" />
        ))}
      </svg>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          type="button"
          onClick={() => setSteps((s) => Math.min(s + 1, 10))}
          disabled={steps >= 10}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: steps >= 10 ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: steps >= 10 ? 0.5 : 1 }}
        >
          梦境前进一步
        </button>
        <button
          type="button"
          onClick={() => setSteps(0)}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        虚线是完全由 M 自回归生成的轨迹 —— C 的策略训练全程只看这条轨迹,从未调用过真实环境或 V 编码的真实帧。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/ControllerStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORLD_MODELS_SOURCE_PATH } from "../lib/prose";
import { DreamRolloutWidget } from "../widgets/DreamRolloutWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ControllerStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:C(Controller)—— 完全在梦境里训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        C 是一个极小的线性控制器,用进化策略(而非梯度下降)训练,训练时用到的所有"经验"都来自 M 自回归生成的梦境轨迹,不是真实环境交互。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <DreamRolloutWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageWorldModels.module.css`**

原样复制"关键背景"里的模板。

- [ ] **Step 4: 写 `NodePageWorldModels.tsx`**

```tsx
import { Link } from "react-router";
import worldModelsMarkdown from "../../../../../../16-world-models/01-world-models.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WORLD_MODELS_SOURCE_PATH } from "./lib/prose";
import { VisionStage } from "./stages/VisionStage";
import { MemoryStage } from "./stages/MemoryStage";
import { ControllerStage } from "./stages/ControllerStage";
import styles from "./NodePageWorldModels.module.css";

const prose = extractProse(worldModelsMarkdown);

export default function NodePageWorldModels() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>World Models (2018)</h1>
        <div className={styles.metaLine}>作者:David Ha · Jürgen Schmidhuber</div>
        <div className={styles.metaLine}>论文:World Models</div>
        <p className={styles.keyIdea}>
          把智能体拆成 V(VAE 视觉压缩)+ M(MDN-RNN 时序预测)+ C(极小线性控制器)三部分,C 完全在 M 生成的"梦境"里训练
        </p>
      </section>

      <section className={styles.stage}>
        <VisionStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MemoryStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ControllerStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WORLD_MODELS_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WORLD_MODELS_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "16-world-models/01-world-models": lazy(() => import("./world-models/NodePageWorldModels")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

Expected: 全部通过,新增 `16-world-models/01-world-models` 冒烟测试用例 PASS,prose 完整性用例 PASS。

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/world-models web/src/components/node/golden/index.ts
git commit -m "feat: World Models 金标本 Stage3 + NodePage + 注册"
```

---

## Task 3: Video Diffusion Models — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/video-diffusion-models/lib/data.ts`
- Create: `web/src/components/node/golden/video-diffusion-models/lib/prose.ts`
- Create: `web/src/components/node/golden/video-diffusion-models/stages/Stage.module.css`
- Create: `web/src/components/node/golden/video-diffusion-models/widgets/FlopsCompareWidget.tsx`
- Create: `web/src/components/node/golden/video-diffusion-models/stages/FactorizedStage.tsx`
- Create: `web/src/components/node/golden/video-diffusion-models/widgets/JointTrainingWidget.tsx`
- Create: `web/src/components/node/golden/video-diffusion-models/stages/JointTrainingStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## \|^### ' 16-world-models/02-video-diffusion-models.md
```

Expected:结构与 World Models 节点一致(双层),机制标题为 `## 机制一:时空分解架构 —— 2D 空间卷积 + 1D 时间卷积`、`## 机制二:图像/视频联合训练`、`## 机制三:条件生成的引导技术 —— reconstruction guidance 与自回归扩展长度`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// Video Diffusion Models demo 数据:2D+1D 分解卷积 vs 3D 卷积的
// FLOPs 量级对比 + 图像/视频联合训练混合比例演示 + 自回归滑窗扩展。

/** 给定分辨率 H×W、帧数 T、卷积核大小 k、通道数 C,估算 3D 卷积的 FLOPs(简化量级公式) */
export function flops3D(h: number, w: number, t: number, k: number, c: number): number {
  return h * w * t * k * k * k * c * c;
}

/** 2D 空间卷积(每帧独立)+ 1D 时间卷积的 FLOPs 之和 */
export function flopsFactorized(h: number, w: number, t: number, k: number, c: number): number {
  const spatial = h * w * t * k * k * c * c; // 2D 卷积对每一帧做
  const temporal = h * w * t * k * c * c; // 1D 卷积沿时间轴
  return spatial + temporal;
}

/** 给定图像:视频混合比例(0=纯视频,1=纯图像),模拟训练 loss 曲线的抖动幅度——
 * 纯视频数据量小,loss 抖动大;混入图像数据后由于数据量大大增加,曲线更平滑。
 * 返回 20 个点的 loss 值(确定性,不是真实训练,仅用于示意趋势)。 */
export function simulateLossCurve(imageRatio: number): number[] {
  const points: number[] = [];
  const noiseScale = 0.3 * (1 - imageRatio) + 0.02;
  for (let i = 0; i < 20; i++) {
    const base = 1.0 * Math.exp(-i / 8) + 0.1;
    let h = (i * 977 + Math.round(imageRatio * 1000) * 31) >>> 0;
    h = (h * 2654435761) >>> 0;
    const noise = (((h % 1000) / 1000) - 0.5) * 2 * noiseScale;
    points.push(Math.max(0.05, base + noise));
  }
  return points;
}

/** 自回归滑窗扩展:给定总窗口大小 windowSize,已生成帧数 generatedCount,
 * 返回当前窗口覆盖的帧区间 [start, end) —— 模拟"用后半窗口的已生成帧作为条件,
 * 继续生成下一窗口"这一自回归扩展长度的过程。 */
export function slidingWindow(windowSize: number, generatedCount: number): { start: number; end: number } {
  if (generatedCount <= windowSize) return { start: 0, end: generatedCount };
  const overlap = Math.floor(windowSize / 2);
  const start = generatedCount - windowSize + overlap - overlap; // 简化:窗口紧跟在已生成序列末尾
  return { start: Math.max(0, generatedCount - windowSize), end: generatedCount };
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用双层模板,`VDM_SOURCE_PATH = "16-world-models/02-video-diffusion-models.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/FlopsCompareWidget.tsx`**

```tsx
import { useState } from "react";
import { flops3D, flopsFactorized } from "../lib/data";

const W = 680;
const H = 300;

export function FlopsCompareWidget() {
  const [resolution, setResolution] = useState(64);
  const frames = 16, kernel = 3, channels = 64;

  const f3d = flops3D(resolution, resolution, frames, kernel, channels);
  const ffact = flopsFactorized(resolution, resolution, frames, kernel, channels);
  const maxF = Math.max(f3d, ffact);

  const bar = (x: number, val: number, label: string, color: string) => {
    const h = Math.min((val / maxF) * (H - 100), H - 100);
    return (
      <g key={label}>
        <rect x={x} y={H - 50 - h} width={100} height={h} fill={color} rx={3} />
        <text x={x + 50} y={H - 50 - h - 8} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
          {(val / 1e9).toFixed(1)}G
        </text>
        <text x={x + 50} y={H - 30} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
          {label}
        </text>
      </g>
    );
  };

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        分辨率 = {resolution}×{resolution}
        <input type="range" min={32} max={128} step={16} value={resolution} onChange={(e) => setResolution(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="3D 卷积与时空分解卷积的算力对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          FLOPs 量级对比(16 帧,{resolution}×{resolution},估算值)
        </text>
        <line x1={30} y1={H - 50} x2={W - 30} y2={H - 50} stroke="var(--border)" />
        {bar(150, f3d, "完整 3D 卷积", "#9ca3af")}
        {bar(400, ffact, "2D+1D 分解", "#d946ef")}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        分辨率越高,3D 卷积的算力开销增长越快;2D+1D 分解把空间和时间维度拆开卷积,复用图像领域已经很成熟的 2D 卷积效率。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/FactorizedStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VDM_SOURCE_PATH } from "../lib/prose";
import { FlopsCompareWidget } from "../widgets/FlopsCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function FactorizedStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:时空分解架构
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用 2D 空间卷积(逐帧)+ 1D 时间卷积(沿时间轴)替代昂贵的完整 3D 卷积,大幅降低视频生成的算力开销。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={VDM_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={VDM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <FlopsCompareWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/JointTrainingWidget.tsx`**

```tsx
import { useState } from "react";
import { simulateLossCurve } from "../lib/data";

const W = 680;
const H = 260;

export function JointTrainingWidget() {
  const [imageRatio, setImageRatio] = useState(0.3);
  const curve = simulateLossCurve(imageRatio);
  const maxL = Math.max(...curve);

  const toX = (i: number) => 40 + (i / (curve.length - 1)) * (W - 80);
  const toY = (v: number) => H - 40 - Math.min((v / maxL) * (H - 80), H - 80);
  const path = curve.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        图像数据占比 = {Math.round(imageRatio * 100)}%(0% = 纯视频 batch,100% = 纯图像 batch)
        <input type="range" min={0} max={100} value={Math.round(imageRatio * 100)} onChange={(e) => setImageRatio(Number(e.target.value) / 100)} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`图像占比 ${Math.round(imageRatio * 100)}% 时的训练 loss 曲线示意`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          训练 loss 曲线示意(混合比例影响抖动幅度)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        纯视频数据量小(标注/采集成本高),loss 曲线抖动大;混入大规模图像数据后曲线更平滑 —— 图像/视频联合训练复用了图像领域的数据规模优势。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/JointTrainingStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VDM_SOURCE_PATH } from "../lib/prose";
import { JointTrainingWidget } from "../widgets/JointTrainingWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function JointTrainingStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:图像/视频联合训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        视频数据采集/标注成本远高于图像,联合训练让模型同时从大规模图像数据集和相对稀缺的视频数据集里学习。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={VDM_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <JointTrainingWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 10: 提交**

```bash
git add web/src/components/node/golden/video-diffusion-models/lib web/src/components/node/golden/video-diffusion-models/stages web/src/components/node/golden/video-diffusion-models/widgets
git commit -m "feat: Video Diffusion Models 金标本 lib + Stage1/2"
```

---

## Task 4: Video Diffusion Models — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/video-diffusion-models/widgets/SlidingWindowWidget.tsx`
- Create: `web/src/components/node/golden/video-diffusion-models/stages/ExtensionStage.tsx`
- Create: `web/src/components/node/golden/video-diffusion-models/NodePageVideoDiffusionModels.tsx`
- Create: `web/src/components/node/golden/video-diffusion-models/NodePageVideoDiffusionModels.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/SlidingWindowWidget.tsx`**

```tsx
import { useState } from "react";
import { slidingWindow } from "../lib/data";

const WINDOW_SIZE = 8;
const TOTAL_CELLS = 24;

// 滑动窗口自回归扩展:每点一次"生成下一窗口",总帧数增加,
// 窗口覆盖的区间随之前移 —— 展示 reconstruction guidance 如何
// 用已生成帧作条件继续扩展视频长度。

export function SlidingWindowWidget() {
  const [generated, setGenerated] = useState(WINDOW_SIZE);
  const { start, end } = slidingWindow(WINDOW_SIZE, generated);

  const cellW = 24;

  return (
    <div>
      <svg viewBox={`0 0 ${TOTAL_CELLS * cellW + 20} 80`} style={{ width: "100%", height: "auto" }} role="img" aria-label={`已生成 ${generated} 帧,当前窗口覆盖 ${start} 到 ${end}`}>
        {Array.from({ length: TOTAL_CELLS }, (_, i) => {
          const inGenerated = i < generated;
          const inWindow = i >= start && i < end;
          const fill = inWindow ? "#d946ef" : inGenerated ? "#f5d0fe" : "var(--bg-subtle)";
          return <rect key={i} x={10 + i * cellW} y={20} width={cellW - 2} height={30} fill={fill} rx={2} />;
        })}
        <text x={10} y={70} fontSize={10} fill="var(--ink-muted)">已生成 {generated} / {TOTAL_CELLS} 帧,当前窗口 [{start}, {end})</text>
      </svg>
      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-2)" }}>
        <button
          type="button"
          onClick={() => setGenerated((g) => Math.min(g + 4, TOTAL_CELLS))}
          disabled={generated >= TOTAL_CELLS}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: generated >= TOTAL_CELLS ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: generated >= TOTAL_CELLS ? 0.5 : 1 }}
        >
          生成下一窗口
        </button>
        <button
          type="button"
          onClick={() => setGenerated(WINDOW_SIZE)}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        深粉色 = 当前生成窗口;浅粉色 = 之前已生成、现在作为条件的帧。窗口不断前移,视频长度随之延长。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/ExtensionStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VDM_SOURCE_PATH } from "../lib/prose";
import { SlidingWindowWidget } from "../widgets/SlidingWindowWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ExtensionStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:条件生成的引导技术
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        reconstruction guidance 配合自回归滑窗,让模型能生成超出单次训练窗口长度的更长视频。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={VDM_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={VDM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SlidingWindowWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageVideoDiffusionModels.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageVideoDiffusionModels.tsx`**

```tsx
import { Link } from "react-router";
import vdmMarkdown from "../../../../../../16-world-models/02-video-diffusion-models.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, VDM_SOURCE_PATH } from "./lib/prose";
import { FactorizedStage } from "./stages/FactorizedStage";
import { JointTrainingStage } from "./stages/JointTrainingStage";
import { ExtensionStage } from "./stages/ExtensionStage";
import styles from "./NodePageVideoDiffusionModels.module.css";

const prose = extractProse(vdmMarkdown);

export default function NodePageVideoDiffusionModels() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>Video Diffusion Models (2022)</h1>
        <div className={styles.metaLine}>作者:Jonathan Ho · Tim Salimans · Alexey Gritsenko · William Chan · Mohammad Norouzi · David J. Fleet</div>
        <div className={styles.metaLine}>论文:Video Diffusion Models</div>
        <p className={styles.keyIdea}>
          把 DDPM 的去噪框架从图像推广到视频:时空分解卷积代替昂贵的 3D 卷积,图像/视频联合训练复用大规模图像数据
        </p>
      </section>

      <section className={styles.stage}>
        <FactorizedStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <JointTrainingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ExtensionStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={VDM_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={VDM_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={VDM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "16-world-models/02-video-diffusion-models": lazy(() => import("./video-diffusion-models/NodePageVideoDiffusionModels")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/video-diffusion-models web/src/components/node/golden/index.ts
git commit -m "feat: Video Diffusion Models 金标本 Stage3 + NodePage + 注册"
```

---

## Task 5: DreamerV3 — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/dreamerv3/lib/data.ts`
- Create: `web/src/components/node/golden/dreamerv3/lib/prose.ts`
- Create: `web/src/components/node/golden/dreamerv3/stages/Stage.module.css`
- Create: `web/src/components/node/golden/dreamerv3/widgets/CategoricalLatentWidget.tsx`
- Create: `web/src/components/node/golden/dreamerv3/stages/RssmStage.tsx`
- Create: `web/src/components/node/golden/dreamerv3/widgets/SymlogWidget.tsx`
- Create: `web/src/components/node/golden/dreamerv3/stages/SymlogStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## \|^### ' 16-world-models/03-dreamerv3.md
```

Expected:双层结构,机制标题 `## 机制一:RSSM 世界模型 —— 离散隐变量`、`## 机制二:symlog 归一化 —— 跨尺度稳定训练`、`## 机制三:纯 latent imagination 训练 actor-critic`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// DreamerV3 demo 数据:离散类别隐变量(RSSM)采样可视化 +
// symlog 归一化跨量级 reward 演示 + 纯想象轨迹 rollout。

/** 用确定性 hash 给 numCategoricals 个类别分布(每个 numClasses 类)生成概率, 模拟 RSSM 离散隐状态 */
export function categoricalLatent(numCategoricals: number, numClasses: number, seed = 0): number[][] {
  const dists: number[][] = [];
  for (let c = 0; c < numCategoricals; c++) {
    const logits: number[] = [];
    for (let k = 0; k < numClasses; k++) {
      let h = (seed + c * 977 + k * 31) >>> 0;
      h = (h * 2654435761) >>> 0;
      logits.push(((h % 1000) / 1000) * 4 - 2);
    }
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const s = exps.reduce((a, b) => a + b, 0);
    dists.push(exps.map((e) => e / s));
  }
  return dists;
}

/** symlog(x) = sign(x) * log(1 + |x|) —— 把跨越多个数量级的数值压缩到可比范围 */
export function symlog(x: number): number {
  return Math.sign(x) * Math.log(1 + Math.abs(x));
}

/** symlog 的反函数,用于从压缩空间还原 */
export function symexp(x: number): number {
  return Math.sign(x) * (Math.exp(Math.abs(x)) - 1);
}

export const DOMAIN_REWARDS: Array<{ domain: string; raw: number }> = [
  { domain: "Atari(单步得分)", raw: 1 },
  { domain: "DMC(连续控制)", raw: 12 },
  { domain: "Minecraft(采集里程碑)", raw: 500 },
  { domain: "稀疏大额奖励", raw: 8000 },
];

/** 纯想象(imagination)轨迹:actor-critic 只在这条轨迹上训练,不接触真实环境 */
export function imaginationRollout(steps: number): Array<{ x: number; y: number }> {
  const traj: Array<{ x: number; y: number }> = [{ x: 0, y: 0 }];
  for (let t = 1; t <= steps; t++) {
    let h = (t * 2654435761) >>> 0;
    const angle = ((h % 1000) / 1000) * Math.PI * 2;
    const prev = traj[traj.length - 1];
    traj.push({ x: prev.x + Math.cos(angle) * 0.6, y: prev.y + Math.sin(angle) * 0.6 });
  }
  return traj;
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用双层模板,`DREAMERV3_SOURCE_PATH = "16-world-models/03-dreamerv3.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/CategoricalLatentWidget.tsx`**

```tsx
import { useState } from "react";
import { categoricalLatent } from "../lib/data";

export function CategoricalLatentWidget() {
  const [numCategoricals, setNumCategoricals] = useState(4);
  const numClasses = 8;
  const dists = categoricalLatent(numCategoricals, numClasses);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        类别变量组数 = {numCategoricals}(每组 {numClasses} 类)
        <input type="range" min={1} max={8} value={numCategoricals} onChange={(e) => setNumCategoricals(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {dists.map((dist, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 50, fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>组 {i}</span>
            <div style={{ display: "flex", gap: 1, flex: 1 }}>
              {dist.map((p, k) => (
                <div key={k} title={p.toFixed(2)} style={{ height: 20, flex: 1, background: "#d946ef", opacity: 0.2 + p * 3 }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        每组是一个 {numClasses} 类的离散分布(颜色深浅表示概率)。组数越多,隐状态能表达的组合数越多(numClasses^numCategoricals),表达力呈指数增长。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/RssmStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DREAMERV3_SOURCE_PATH } from "../lib/prose";
import { CategoricalLatentWidget } from "../widgets/CategoricalLatentWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function RssmStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:RSSM 世界模型 —— 离散隐变量
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用多组离散类别分布(而非连续高斯)表示隐状态,组合数随组数指数增长,能表达更丰富的不确定性和多模态未来。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <CategoricalLatentWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/SymlogWidget.tsx`**

```tsx
import { DOMAIN_REWARDS, symlog } from "../lib/data";

const W = 680;
const H = 300;

export function SymlogWidget() {
  const maxRaw = Math.max(...DOMAIN_REWARDS.map((d) => d.raw));
  const maxSymlog = Math.max(...DOMAIN_REWARDS.map((d) => symlog(d.raw)));

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="不同领域原始 reward 与 symlog 变换后的对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          跨领域 reward 量级 vs symlog 归一化后
        </text>
        {DOMAIN_REWARDS.map((d, i) => {
          const y = 50 + i * 60;
          const rawW = Math.min((d.raw / maxRaw) * 200, 200);
          const symW = Math.min((symlog(d.raw) / maxSymlog) * 200, 200);
          return (
            <g key={d.domain}>
              <text x={10} y={y + 4} fontSize={11} fill="var(--ink-secondary)">{d.domain}</text>
              <rect x={220} y={y - 8} width={rawW} height={7} fill="#9ca3af" />
              <text x={220 + rawW + 4} y={y - 2} fontSize={9} fill="var(--ink-muted)">raw={d.raw}</text>
              <rect x={220} y={y + 3} width={symW} height={7} fill="#d946ef" />
              <text x={220 + symW + 4} y={y + 9} fontSize={9} fill="var(--ink-muted)">symlog={symlog(d.raw).toFixed(2)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        灰条 = 原始 reward(跨越 4 个数量级);粉条 = symlog 变换后,全部被压缩到相近的可比范围 —— 同一套超参数因此能跨领域通用,不需要为每个领域单独调 reward scale。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/SymlogStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DREAMERV3_SOURCE_PATH } from "../lib/prose";
import { SymlogWidget } from "../widgets/SymlogWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function SymlogStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:symlog 归一化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        不同任务领域的 reward/输出值天差地别,symlog 把它们统一压缩到可比范围,是固定同一套超参数跨领域通吃的关键工程细节。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DREAMERV3_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <SymlogWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 10: 提交**

```bash
git add web/src/components/node/golden/dreamerv3/lib web/src/components/node/golden/dreamerv3/stages web/src/components/node/golden/dreamerv3/widgets
git commit -m "feat: DreamerV3 金标本 lib + Stage1/2"
```

---

## Task 6: DreamerV3 — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/dreamerv3/widgets/ImaginationRolloutWidget.tsx`
- Create: `web/src/components/node/golden/dreamerv3/stages/ImaginationStage.tsx`
- Create: `web/src/components/node/golden/dreamerv3/NodePageDreamerV3.tsx`
- Create: `web/src/components/node/golden/dreamerv3/NodePageDreamerV3.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/ImaginationRolloutWidget.tsx`**

```tsx
import { useState } from "react";
import { imaginationRollout } from "../lib/data";

const W = 500;
const H = 300;

export function ImaginationRolloutWidget() {
  const [steps, setSteps] = useState(5);
  const traj = imaginationRollout(steps);

  const xs = traj.map((p) => p.x);
  const ys = traj.map((p) => p.y);
  const minX = Math.min(...xs, -1), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1), maxY = Math.max(...ys, 1);
  const toX = (x: number) => 30 + ((x - minX) / (maxX - minX)) * (W - 60);
  const toY = (y: number) => 30 + ((y - minY) / (maxY - minY)) * (H - 60);

  const path = traj.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.x)} ${toY(p.y)}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        想象步数 = {steps}
        <input type="range" min={1} max={15} value={steps} onChange={(e) => setSteps(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`纯想象轨迹,${steps} 步`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          actor-critic 训练用的纯想象轨迹(不接触真实环境)
        </text>
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} strokeDasharray="4 2" />
        {traj.map((p, i) => (
          <circle key={i} cx={toX(p.x)} cy={toY(p.y)} r={i === 0 ? 6 : 4} fill={i === 0 ? "#9d174d" : "#d946ef"} />
        ))}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        深色起点是当前真实状态编码,之后每一步都由世界模型在隐空间里自回归展开——actor 和 critic 的梯度全部来自这条虚拟轨迹。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/ImaginationStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DREAMERV3_SOURCE_PATH } from "../lib/prose";
import { ImaginationRolloutWidget } from "../widgets/ImaginationRolloutWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ImaginationStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:纯 latent imagination 训练 actor-critic
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        actor 和 critic 完全在世界模型生成的想象轨迹上训练,不需要额外调用真实环境采样,大幅提升样本效率。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <ImaginationRolloutWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageDreamerV3.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageDreamerV3.tsx`**

```tsx
import { Link } from "react-router";
import dreamerv3Markdown from "../../../../../../16-world-models/03-dreamerv3.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DREAMERV3_SOURCE_PATH } from "./lib/prose";
import { RssmStage } from "./stages/RssmStage";
import { SymlogStage } from "./stages/SymlogStage";
import { ImaginationStage } from "./stages/ImaginationStage";
import styles from "./NodePageDreamerV3.module.css";

const prose = extractProse(dreamerv3Markdown);

export default function NodePageDreamerV3() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>DreamerV3 (2023)</h1>
        <div className={styles.metaLine}>作者:Danijar Hafner · Jurgis Pasukonis · Jimmy Ba · Timothy Lillicrap</div>
        <div className={styles.metaLine}>论文:Mastering Diverse Domains through World Models</div>
        <p className={styles.keyIdea}>
          把 latent imagination 式的 model-based RL 规模化到跨领域通吃,固定同一套超参数不调参就能匹配甚至超过各领域的 model-free SOTA
        </p>
      </section>

      <section className={styles.stage}>
        <RssmStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <SymlogStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ImaginationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DREAMERV3_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DREAMERV3_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "16-world-models/03-dreamerv3": lazy(() => import("./dreamerv3/NodePageDreamerV3")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/dreamerv3 web/src/components/node/golden/index.ts
git commit -m "feat: DreamerV3 金标本 Stage3 + NodePage + 注册"
```

---

## Task 7: Sora — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/sora/lib/data.ts`
- Create: `web/src/components/node/golden/sora/lib/prose.ts`
- Create: `web/src/components/node/golden/sora/stages/Stage.module.css`
- Create: `web/src/components/node/golden/sora/widgets/SpacetimePatchWidget.tsx`
- Create: `web/src/components/node/golden/sora/stages/PatchifyStage.tsx`
- Create: `web/src/components/node/golden/sora/widgets/ScaleQualityWidget.tsx`
- Create: `web/src/components/node/golden/sora/stages/ScalingStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## \|^### ' 16-world-models/04-sora.md
```

Expected:双层结构,机制标题 `## 机制一:视频压缩网络 + spacetime patches`、`## 机制二:Diffusion Transformer 主干规模化到视频`、`## 机制三:原生分辨率/长宽比/时长训练`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// Sora demo 数据:spacetime patch 切分可视化 + 模型规模 vs 质量曲线 +
// 原生分辨率/时长预设的 patch 数量演示。

export interface VideoConfig {
  label: string;
  frames: number;
  height: number;
  width: number;
}

export const PRESET_CONFIGS: VideoConfig[] = [
  { label: "方形短片", frames: 8, height: 4, width: 4 },
  { label: "竖屏(9:16)", frames: 12, height: 6, width: 3 },
  { label: "宽屏(16:9)", frames: 6, height: 3, width: 6 },
  { label: "长视频", frames: 20, height: 4, width: 4 },
];

/** 给定视频体和 patch 大小,计算沿三个维度的 patch 数量 */
export function patchCounts(cfg: VideoConfig, patchSize: number): { pt: number; ph: number; pw: number; total: number } {
  const pt = Math.ceil(cfg.frames / patchSize);
  const ph = Math.ceil(cfg.height / patchSize);
  const pw = Math.ceil(cfg.width / patchSize);
  return { pt, ph, pw, total: pt * ph * pw };
}

/** 模型规模(参数量档位,单位任意)vs 生成质量的示意曲线:边际收益递减但不封顶 */
export function scaleToQuality(scale: number): number {
  return 1 - Math.exp(-scale / 3);
}

export const SCALE_PRESETS = [0.5, 1, 2, 4, 8, 16];
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用双层模板,`SORA_SOURCE_PATH = "16-world-models/04-sora.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/SpacetimePatchWidget.tsx`**

```tsx
import { useState } from "react";
import { PRESET_CONFIGS, patchCounts } from "../lib/data";

export function SpacetimePatchWidget() {
  const [configIdx, setConfigIdx] = useState(0);
  const [patchSize, setPatchSize] = useState(2);
  const cfg = PRESET_CONFIGS[configIdx];
  const { pt, ph, pw, total } = patchCounts(cfg, patchSize);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {PRESET_CONFIGS.map((c, i) => (
          <button
            key={c.label} type="button" onClick={() => setConfigIdx(i)} aria-pressed={i === configIdx}
            style={{
              padding: "4px 10px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === configIdx ? "#d946ef" : "var(--border)"}`,
              background: i === configIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === configIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {c.label}
          </button>
        ))}
      </div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        patch 大小 = {patchSize}
        <input type="range" min={1} max={4} value={patchSize} onChange={(e) => setPatchSize(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", gap: 2, flexWrap: "wrap", maxWidth: 300 }}>
        {Array.from({ length: total }, (_, i) => (
          <div key={i} style={{ width: 14, height: 14, background: "#d946ef", opacity: 0.4 + (i % 5) * 0.1, borderRadius: 2 }} />
        ))}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        当前配置({cfg.frames} 帧 × {cfg.height}×{cfg.width})切分成 {pt}×{ph}×{pw} = {total} 个时空 patch。不同长宽比/时长的视频都能统一表示成变长的 patch 序列喂给 DiT。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/PatchifyStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SORA_SOURCE_PATH } from "../lib/prose";
import { SpacetimePatchWidget } from "../widgets/SpacetimePatchWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function PatchifyStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:视频压缩网络 + spacetime patches
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把视频压缩到低维时空潜空间后切成 patch 序列,不同长宽比/时长的输入统一表示成变长 token 序列。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SORA_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SORA_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SpacetimePatchWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/ScaleQualityWidget.tsx`**

```tsx
import { useState } from "react";
import { SCALE_PRESETS, scaleToQuality } from "../lib/data";

const W = 500;
const H = 260;

export function ScaleQualityWidget() {
  const [scaleIdx, setScaleIdx] = useState(2);
  const scale = SCALE_PRESETS[scaleIdx];
  const quality = scaleToQuality(scale);

  const toX = (s: number) => 40 + (s / 16) * (W - 80);
  const toY = (q: number) => H - 40 - q * (H - 80);
  const curvePoints = Array.from({ length: 50 }, (_, i) => (i / 49) * 16);
  const path = curvePoints.map((s, i) => `${i === 0 ? "M" : "L"} ${toX(s)} ${toY(scaleToQuality(s))}`).join(" ");

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {SCALE_PRESETS.map((s, i) => (
          <button
            key={s} type="button" onClick={() => setScaleIdx(i)} aria-pressed={i === scaleIdx}
            style={{
              padding: "4px 10px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === scaleIdx ? "#d946ef" : "var(--border)"}`,
              background: i === scaleIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === scaleIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {s}×
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`模型规模 ${scale} 倍时的生成质量示意`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          DiT 规模化:模型规模 vs 生成质量(示意曲线)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
        <circle cx={toX(scale)} cy={toY(quality)} r={6} fill="#9d174d" />
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        与 DiT 论文一致的规模化规律:算力/参数量越大,质量持续提升但边际收益递减,没有观察到饱和天花板。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/ScalingStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SORA_SOURCE_PATH } from "../lib/prose";
import { ScaleQualityWidget } from "../widgets/ScaleQualityWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ScalingStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Diffusion Transformer 主干规模化到视频
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        延续 DiT 的规模化规律,把同一套 Transformer 主干直接放大到视频这个更高维的数据模态上。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SORA_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <ScaleQualityWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 10: 提交**

```bash
git add web/src/components/node/golden/sora/lib web/src/components/node/golden/sora/stages web/src/components/node/golden/sora/widgets
git commit -m "feat: Sora 金标本 lib + Stage1/2"
```

---

## Task 8: Sora — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/sora/widgets/NativeResolutionWidget.tsx`
- Create: `web/src/components/node/golden/sora/stages/NativeResolutionStage.tsx`
- Create: `web/src/components/node/golden/sora/NodePageSora.tsx`
- Create: `web/src/components/node/golden/sora/NodePageSora.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/NativeResolutionWidget.tsx`**

```tsx
import { useState } from "react";
import { PRESET_CONFIGS, patchCounts } from "../lib/data";

// 同一个模型不改架构,直接切换不同长宽比/时长预设,展示 patch
// 数量如何自适应变化 —— 不需要为每种分辨率单独训练/裁剪。

export function NativeResolutionWidget() {
  const [configIdx, setConfigIdx] = useState(0);
  const patchSize = 2;

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
        {PRESET_CONFIGS.map((c, i) => (
          <button
            key={c.label} type="button" onClick={() => setConfigIdx(i)} aria-pressed={i === configIdx}
            style={{
              padding: "4px 10px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === configIdx ? "#d946ef" : "var(--border)"}`,
              background: i === configIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === configIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {c.label}
          </button>
        ))}
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>配置</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>帧×高×宽</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>patch 数</th>
          </tr>
        </thead>
        <tbody>
          {PRESET_CONFIGS.map((c, i) => {
            const { total } = patchCounts(c, patchSize);
            const active = i === configIdx;
            return (
              <tr key={c.label} style={{ background: active ? "#fce7f3" : "transparent" }}>
                <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", fontWeight: active ? 700 : 400 }}>{c.label}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", fontSize: "var(--fs-sm)" }}>{c.frames}×{c.height}×{c.width}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", fontSize: "var(--fs-sm)" }}>{total}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        同一套模型架构,不需要为每种长宽比/时长单独裁剪或重训 —— patch 序列长度自动适配输入尺寸。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/NativeResolutionStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SORA_SOURCE_PATH } from "../lib/prose";
import { NativeResolutionWidget } from "../widgets/NativeResolutionWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function NativeResolutionStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:原生分辨率/长宽比/时长训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        不强制裁剪/缩放到统一尺寸,直接在原生分辨率/长宽比/时长上训练,变长 patch 序列天然支持这种灵活性。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SORA_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SORA_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <NativeResolutionWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageSora.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageSora.tsx`**

```tsx
import { Link } from "react-router";
import soraMarkdown from "../../../../../../16-world-models/04-sora.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SORA_SOURCE_PATH } from "./lib/prose";
import { PatchifyStage } from "./stages/PatchifyStage";
import { ScalingStage } from "./stages/ScalingStage";
import { NativeResolutionStage } from "./stages/NativeResolutionStage";
import styles from "./NodePageSora.module.css";

const prose = extractProse(soraMarkdown);

export default function NodePageSora() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>Sora (2024)</h1>
        <div className={styles.metaLine}>作者:OpenAI</div>
        <div className={styles.metaLine}>论文:Video generation models as world simulators</div>
        <p className={styles.keyIdea}>
          把 DiT 规模化到分钟级、多分辨率、多时长连贯视频:用 spacetime patches 统一表示不同长宽比/时长的时空数据
        </p>
      </section>

      <section className={styles.stage}>
        <PatchifyStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ScalingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <NativeResolutionStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SORA_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={SORA_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SORA_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "16-world-models/04-sora": lazy(() => import("./sora/NodePageSora")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/sora web/src/components/node/golden/index.ts
git commit -m "feat: Sora 金标本 Stage3 + NodePage + 注册"
```

---

## Task 9: Genie — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/genie/lib/data.ts`
- Create: `web/src/components/node/golden/genie/lib/prose.ts`
- Create: `web/src/components/node/golden/genie/stages/Stage.module.css`
- Create: `web/src/components/node/golden/genie/widgets/TokenizerWidget.tsx`
- Create: `web/src/components/node/golden/genie/stages/TokenizerStage.tsx`
- Create: `web/src/components/node/golden/genie/widgets/LamWidget.tsx`
- Create: `web/src/components/node/golden/genie/stages/LamStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## \|^### ' 16-world-models/05-genie.md
```

Expected:双层结构,机制标题 `## 机制一:视频 tokenizer —— 把原始帧压缩成离散 token 序列`、`## 机制二:Latent Action Model(LAM)—— 无监督推断离散动作空间`、`## 机制三:动态模型 —— 给定 latent action 自回归生成下一帧`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// Genie demo 数据:帧 → 离散 token(tokenizer)+ 相邻帧推断离散动作(LAM)+
// 给定动作自回归生成下一帧(动态模型)。全部确定性 hash,不真跑训练。

export const NUM_ACTIONS = 8;
export const GRID_SIZE = 6;

/** 用简单参数化生成一个 toy "帧":以 (cx, cy) 为中心的圆形亮斑 */
export function makeFrame(cx: number, cy: number): number[] {
  return Array.from({ length: GRID_SIZE * GRID_SIZE }, (_, i) => {
    const x = i % GRID_SIZE, y = Math.floor(i / GRID_SIZE);
    const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
    return Math.max(0, 1 - d / 2.5);
  });
}

export const INITIAL_FRAME = { cx: 2.5, cy: 2.5 };

/** tokenizer:把帧(64 维)哈希成一个离散 token id(0-255) */
export function tokenizeFrame(frame: number[]): number {
  let h = 0;
  for (let i = 0; i < frame.length; i++) {
    h = (h * 31 + Math.round(frame[i] * 100)) >>> 0;
  }
  return h % 256;
}

/** LAM:给定相邻两帧的中心位移,无监督推断出的离散动作 id(0-7,8 个方向) */
export function inferLatentAction(prevPos: { cx: number; cy: number }, currPos: { cx: number; cy: number }): number {
  const dx = currPos.cx - prevPos.cx;
  const dy = currPos.cy - prevPos.cy;
  if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return -1; // 无动作
  const angle = Math.atan2(dy, dx);
  const idx = Math.round(((angle + Math.PI) / (2 * Math.PI)) * NUM_ACTIONS) % NUM_ACTIONS;
  return idx;
}

export const ACTION_VECTORS: Array<{ dx: number; dy: number; label: string }> = Array.from({ length: NUM_ACTIONS }, (_, i) => {
  const angle = (i / NUM_ACTIONS) * 2 * Math.PI - Math.PI;
  return { dx: Math.cos(angle), dy: Math.sin(angle), label: `动作${i}` };
});

/** 动态模型:给定当前位置 + 选择的动作 id,自回归生成下一帧的位置(确定性物理规则模拟) */
export function dynamicsStep(pos: { cx: number; cy: number }, actionId: number): { cx: number; cy: number } {
  const v = ACTION_VECTORS[actionId];
  const nx = Math.max(0, Math.min(GRID_SIZE - 1, pos.cx + v.dx * 0.8));
  const ny = Math.max(0, Math.min(GRID_SIZE - 1, pos.cy + v.dy * 0.8));
  return { cx: nx, cy: ny };
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用双层模板,`GENIE_SOURCE_PATH = "16-world-models/05-genie.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/TokenizerWidget.tsx`**

```tsx
import { useState } from "react";
import { GRID_SIZE, INITIAL_FRAME, makeFrame, tokenizeFrame } from "../lib/data";

function GridSvg({ values, size = 140 }: { values: number[]; size?: number }) {
  const cell = size / GRID_SIZE;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {values.map((v, i) => {
        const x = (i % GRID_SIZE) * cell;
        const y = Math.floor(i / GRID_SIZE) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${Math.round(g * 0.6)},${g})`} stroke="var(--bg-canvas)" strokeWidth={0.5} />;
      })}
    </svg>
  );
}

export function TokenizerWidget() {
  const [cx, setCx] = useState(INITIAL_FRAME.cx);
  const frame = makeFrame(cx, INITIAL_FRAME.cy);
  const token = tokenizeFrame(frame);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        移动亮斑位置(模拟不同帧)
        <input type="range" min={0} max={5} step={0.5} value={cx} onChange={(e) => setCx(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", gap: "var(--space-4)", alignItems: "center" }}>
        <GridSvg values={frame} />
        <div style={{ fontSize: "var(--fs-2xl)", color: "var(--ink-muted)" }}>→</div>
        <div style={{ padding: "var(--space-4)", border: "1px solid #d946ef", borderRadius: "var(--radius-md)", background: "#fae8ff" }}>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>token id</div>
          <div style={{ fontSize: "var(--fs-2xl)", fontWeight: 700, color: "#86198f" }}>{token}</div>
        </div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        tokenizer 把每一帧压缩成一个离散 token,后续 LAM 和动态模型全部在 token 序列上工作,不直接处理像素。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/TokenizerStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GENIE_SOURCE_PATH } from "../lib/prose";
import { TokenizerWidget } from "../widgets/TokenizerWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function TokenizerStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:视频 tokenizer
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把原始帧压缩成离散 token 序列,是后续无监督推断动作、自回归生成下一帧的共同基础表示。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GENIE_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GENIE_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <TokenizerWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/LamWidget.tsx`**

```tsx
import { useState } from "react";
import { INITIAL_FRAME, ACTION_VECTORS, inferLatentAction, makeFrame } from "../lib/data";

function MiniFrame({ cx, cy }: { cx: number; cy: number }) {
  const frame = makeFrame(cx, cy);
  const size = 90, grid = 6, cell = size / grid;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {frame.map((v, i) => {
        const x = (i % grid) * cell, y = Math.floor(i / grid) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${Math.round(g * 0.6)},${g})`} />;
      })}
    </svg>
  );
}

export function LamWidget() {
  const [targetIdx, setTargetIdx] = useState(0);
  const prev = INITIAL_FRAME;
  const target = ACTION_VECTORS[targetIdx];
  const curr = { cx: prev.cx + target.dx * 0.8, cy: prev.cy + target.dy * 0.8 };
  const inferred = inferLatentAction(prev, curr);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {ACTION_VECTORS.map((a, i) => (
          <button
            key={a.label} type="button" onClick={() => setTargetIdx(i)} aria-pressed={i === targetIdx}
            style={{
              width: 30, height: 30, borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === targetIdx ? "#d946ef" : "var(--border)"}`,
              background: i === targetIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === targetIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {i}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: "var(--space-4)", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>帧 t</div>
          <MiniFrame cx={prev.cx} cy={prev.cy} />
        </div>
        <div style={{ fontSize: "var(--fs-xl)", color: "var(--ink-muted)" }}>→</div>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>帧 t+1</div>
          <MiniFrame cx={curr.cx} cy={curr.cy} />
        </div>
      </div>
      <div style={{ marginTop: "var(--space-3)", padding: "var(--space-3)", border: "1px solid #d946ef", borderRadius: "var(--radius-md)", background: "#fae8ff" }}>
        <span style={{ fontSize: "var(--fs-sm)", color: "#86198f" }}>
          LAM 无监督推断出的 latent action id = <strong>{inferred}</strong>(未使用任何人工动作标注,纯粹从两帧的差异里推断)
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/LamStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GENIE_SOURCE_PATH } from "../lib/prose";
import { LamWidget } from "../widgets/LamWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function LamStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Latent Action Model(LAM)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        从海量无标注视频里,仅凭相邻帧的变化就能无监督推断出一套离散动作空间,不需要任何人工标注的动作数据。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GENIE_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <LamWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 10: 提交**

```bash
git add web/src/components/node/golden/genie/lib web/src/components/node/golden/genie/stages web/src/components/node/golden/genie/widgets
git commit -m "feat: Genie 金标本 lib + Stage1/2"
```

---

## Task 10: Genie — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/genie/widgets/PlayWidget.tsx`
- Create: `web/src/components/node/golden/genie/stages/PlayStage.tsx`
- Create: `web/src/components/node/golden/genie/NodePageGenie.tsx`
- Create: `web/src/components/node/golden/genie/NodePageGenie.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/PlayWidget.tsx`**

```tsx
import { useState } from "react";
import { INITIAL_FRAME, ACTION_VECTORS, dynamicsStep, makeFrame } from "../lib/data";

const GRID = 6;

function FrameSvg({ cx, cy, size = 180 }: { cx: number; cy: number; size?: number }) {
  const frame = makeFrame(cx, cy);
  const cell = size / GRID;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {frame.map((v, i) => {
        const x = (i % GRID) * cell, y = Math.floor(i / GRID) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${Math.round(g * 0.6)},${g})`} stroke="var(--bg-canvas)" strokeWidth={0.5} />;
      })}
    </svg>
  );
}

// 交互式"玩":点击 8 个离散动作按钮之一,动态模型自回归生成下一帧。
// 全程只用 tokenizer/LAM 学到的离散动作空间,不需要连续控制信号。

export function PlayWidget() {
  const [pos, setPos] = useState(INITIAL_FRAME);
  const [history, setHistory] = useState<number[]>([]);

  return (
    <div>
      <FrameSvg cx={pos.cx} cy={pos.cy} />
      <div style={{ display: "flex", gap: 6, marginTop: "var(--space-3)", flexWrap: "wrap" }}>
        {ACTION_VECTORS.map((a, i) => (
          <button
            key={a.label} type="button"
            onClick={() => { setPos((p) => dynamicsStep(p, i)); setHistory((h) => [...h, i]); }}
            style={{
              width: 32, height: 32, borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)", background: "var(--bg-surface)",
              color: "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {i}
          </button>
        ))}
        <button
          type="button" onClick={() => { setPos(INITIAL_FRAME); setHistory([]); }}
          style={{ padding: "4px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-xs)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        已执行动作序列:{history.join(" → ") || "(无)"}。每点一个动作按钮,动态模型就自回归生成下一帧——这就是"用学到的离散动作逐帧玩生成出来的世界"。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/PlayStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GENIE_SOURCE_PATH } from "../lib/prose";
import { PlayWidget } from "../widgets/PlayWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PlayStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:动态模型 —— 给定 latent action 自回归生成下一帧
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        给定当前 token 序列和选择的离散动作,动态模型自回归预测下一帧的 token —— 用户可以用学到的动作逐帧"玩"生成出的世界。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GENIE_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GENIE_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <PlayWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageGenie.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageGenie.tsx`**

```tsx
import { Link } from "react-router";
import genieMarkdown from "../../../../../../16-world-models/05-genie.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GENIE_SOURCE_PATH } from "./lib/prose";
import { TokenizerStage } from "./stages/TokenizerStage";
import { LamStage } from "./stages/LamStage";
import { PlayStage } from "./stages/PlayStage";
import styles from "./NodePageGenie.module.css";

const prose = extractProse(genieMarkdown);

export default function NodePageGenie() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>Genie (2024)</h1>
        <div className={styles.metaLine}>作者:Jake Bruce · Michael Dennis · Ashley Edwards 等(DeepMind)</div>
        <div className={styles.metaLine}>论文:Genie: Generative Interactive Environments</div>
        <p className={styles.keyIdea}>
          无监督地从海量无标注互联网视频里学出逐帧可控制的生成式环境:隐式学习出离散的 latent action 空间
        </p>
      </section>

      <section className={styles.stage}>
        <TokenizerStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LamStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PlayStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GENIE_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GENIE_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GENIE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

**注意**:作者列表在 frontmatter/markdown 正文里已有完整列表(25 位),此处 hero 区域按既有惯例只显示前几位 + "等",实现时读取 `16-world-models/05-genie.md` 的 frontmatter `authors` 数组确认署名格式与其他节点一致的呈现方式(参照 `mixtral`/`gcn` 等节点 hero 区域对多作者的处理方式,若某篇作者数量很多,直接照抄 markdown frontmatter 里的前 3 位 + "等"格式,不需要全部列出)。

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "16-world-models/05-genie": lazy(() => import("./genie/NodePageGenie")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/genie web/src/components/node/golden/index.ts
git commit -m "feat: Genie 金标本 Stage3 + NodePage + 注册"
```

---

## Task 11: GameNGen — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/gamengen/lib/data.ts`
- Create: `web/src/components/node/golden/gamengen/lib/prose.ts`
- Create: `web/src/components/node/golden/gamengen/stages/Stage.module.css`
- Create: `web/src/components/node/golden/gamengen/widgets/RlTrajectoryWidget.tsx`
- Create: `web/src/components/node/golden/gamengen/stages/RlDataStage.tsx`
- Create: `web/src/components/node/golden/gamengen/widgets/NextFramePredictWidget.tsx`
- Create: `web/src/components/node/golden/gamengen/stages/DiffusionPredictStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## \|^### ' 16-world-models/06-gamengen.md
```

Expected:双层结构,机制标题 `## 机制一:RL agent 自动生成训练数据 —— 不靠人类录屏`、`## 机制二:条件 diffusion 模型预测下一帧 —— 替代渲染步骤本身`、`## 机制三:噪声增强条件帧 —— 对抗自回归漂移的核心工程细节`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// GameNGen demo 数据:RL agent 自动生成的轨迹 + 条件 diffusion 预测下一帧 +
// 噪声增强 vs 不增强时的长程自回归漂移对比。全部确定性构造。

export interface TrajectoryStep {
  x: number;
  y: number;
  action: string;
}

/** RL agent 自我博弈生成的一段训练轨迹(确定性,模拟"自动生成而非人类录屏") */
export function generateRlTrajectory(steps: number): TrajectoryStep[] {
  const actions = ["前进", "左转", "右转", "开火"];
  const traj: TrajectoryStep[] = [];
  let x = 0, y = 0;
  for (let t = 0; t < steps; t++) {
    let h = (t * 2654435761) >>> 0;
    const actionIdx = h % actions.length;
    const angle = ((h >>> 8) % 360) * (Math.PI / 180);
    x += Math.cos(angle) * 0.5;
    y += Math.sin(angle) * 0.5;
    traj.push({ x, y, action: actions[actionIdx] });
  }
  return traj;
}

/** 给定历史帧质量分(0-1)和是否加噪声增强,模拟自回归 N 步后的画面质量衰减曲线。
 * 不加噪声增强:训练时从未见过"自己生成的略有瑕疵的帧"作为条件,推理时误差逐步放大(漂移);
 * 加噪声增强:训练时人为在条件帧上加噪声,模型学会了"即使条件帧不完美也能修正",漂移显著变慢。 */
export function driftCurve(withNoiseAugmentation: boolean, steps: number): number[] {
  const curve: number[] = [];
  for (let t = 0; t <= steps; t++) {
    const decayRate = withNoiseAugmentation ? 0.008 : 0.035;
    curve.push(Math.max(0.1, 1 - decayRate * t - (withNoiseAugmentation ? 0 : 0.0006 * t * t)));
  }
  return curve;
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用双层模板,`GAMENGEN_SOURCE_PATH = "16-world-models/06-gamengen.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/RlTrajectoryWidget.tsx`**

```tsx
import { useState } from "react";
import { generateRlTrajectory } from "../lib/data";

const W = 500;
const H = 260;

export function RlTrajectoryWidget() {
  const [steps, setSteps] = useState(10);
  const traj = generateRlTrajectory(steps);

  const xs = traj.map((p) => p.x), ys = traj.map((p) => p.y);
  const minX = Math.min(...xs, -1), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1), maxY = Math.max(...ys, 1);
  const toX = (x: number) => 30 + ((x - minX) / (maxX - minX || 1)) * (W - 60);
  const toY = (y: number) => 30 + ((y - minY) / (maxY - minY || 1)) * (H - 60);
  const path = traj.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.x)} ${toY(p.y)}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        轨迹步数 = {steps}
        <input type="range" min={2} max={30} value={steps} onChange={(e) => setSteps(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`RL agent 自动生成的 ${steps} 步训练轨迹`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          RL agent 自我博弈生成的训练轨迹(替代人类录屏)
        </text>
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
        {traj.length > 0 && <circle cx={toX(traj[traj.length - 1].x)} cy={toY(traj[traj.length - 1].y)} r={5} fill="#9d174d" />}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        最新动作:{traj[traj.length - 1]?.action ?? "-"}。整段(帧,动作)序列全部由 agent 自我博弈自动产生,不依赖任何人类玩家录屏标注。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/RlDataStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAMENGEN_SOURCE_PATH } from "../lib/prose";
import { RlTrajectoryWidget } from "../widgets/RlTrajectoryWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function RlDataStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:RL agent 自动生成训练数据
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用强化学习 agent 自我博弈产出海量(帧,动作)训练对,替代成本高昂的人类录屏采集。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <RlTrajectoryWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/NextFramePredictWidget.tsx`**

```tsx
const HISTORY_LEN = 4;

// 展示条件 diffusion "预测下一帧"这一步替代传统渲染引擎的渲染循环:
// 历史帧 + 动作条件 → diffusion 模型 → 下一帧。静态示意图,不需要交互状态。

export function NextFramePredictWidget() {
  const frames = Array.from({ length: HISTORY_LEN }, (_, i) => i);

  return (
    <div>
      <svg viewBox="0 0 500 200" style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="条件 diffusion 模型预测下一帧的数据流">
        {frames.map((i) => (
          <g key={i}>
            <rect x={20 + i * 70} y={60} width={55} height={55} rx={4} fill="var(--bg-subtle)" stroke="var(--border)" />
            <text x={20 + i * 70 + 27} y={130} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">t-{HISTORY_LEN - i}</text>
          </g>
        ))}
        <text x={310} y={30} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink-primary)">+ 动作条件</text>
        <path d="M 300 90 L 340 90" stroke="var(--ink-muted)" markerEnd="url(#arrow)" strokeWidth={1.5} />
        <rect x={345} y={55} width={70} height={65} rx={6} fill="#fae8ff" stroke="#d946ef" strokeWidth={2} />
        <text x={380} y={92} textAnchor="middle" fontSize={10} fontWeight={700} fill="#86198f">Diffusion</text>
        <path d="M 415 90 L 450 90" stroke="var(--ink-muted)" markerEnd="url(#arrow)" strokeWidth={1.5} />
        <rect x={455} y={62} width={40} height={40} rx={4} fill="#d946ef" />
        <text x={475} y={120} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">帧 t</text>
        <defs>
          <marker id="arrow" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--ink-muted)" />
          </marker>
        </defs>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        近期 {HISTORY_LEN} 帧历史 + 玩家动作输入 → 条件 diffusion 模型 → 直接预测出下一帧画面,整个过程完全替代传统游戏引擎的渲染循环。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/DiffusionPredictStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAMENGEN_SOURCE_PATH } from "../lib/prose";
import { NextFramePredictWidget } from "../widgets/NextFramePredictWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DiffusionPredictStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:条件 diffusion 模型预测下一帧
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        游戏引擎的渲染循环本质上是"给定历史状态和玩家输入,画出下一帧"的函数——diffusion 模型直接学会了这个函数。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GAMENGEN_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <NextFramePredictWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 10: 提交**

```bash
git add web/src/components/node/golden/gamengen/lib web/src/components/node/golden/gamengen/stages web/src/components/node/golden/gamengen/widgets
git commit -m "feat: GameNGen 金标本 lib + Stage1/2"
```

---

## Task 12: GameNGen — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/gamengen/widgets/DriftCompareWidget.tsx`
- Create: `web/src/components/node/golden/gamengen/stages/NoiseAugmentationStage.tsx`
- Create: `web/src/components/node/golden/gamengen/NodePageGameNGen.tsx`
- Create: `web/src/components/node/golden/gamengen/NodePageGameNGen.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/DriftCompareWidget.tsx`**

```tsx
import { useState } from "react";
import { driftCurve } from "../lib/data";

const W = 600;
const H = 280;

// 开关"是否对条件帧加噪声增强",对比长程自回归多步生成后画面质量
// 是否发生漂移退化。这是 GameNGen 论文的核心工程细节。

export function DriftCompareWidget() {
  const [withAug, setWithAug] = useState(false);
  const steps = 60;
  const curveOff = driftCurve(false, steps);
  const curveOn = driftCurve(true, steps);

  const toX = (t: number) => 40 + (t / steps) * (W - 80);
  const toY = (v: number) => H - 40 - v * (H - 80);
  const pathOff = curveOff.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");
  const pathOn = curveOn.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");

  return (
    <div>
      <button
        type="button" onClick={() => setWithAug((v) => !v)} aria-pressed={withAug}
        style={{
          padding: "4px 14px", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)",
          border: `1px solid ${withAug ? "#d946ef" : "var(--border)"}`,
          background: withAug ? "#d946ef" : "var(--bg-surface)",
          color: withAug ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
        }}
      >
        {withAug ? "✓ 已开启噪声增强" : "开启噪声增强"}
      </button>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="噪声增强对长程自回归漂移的影响对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          长程自回归 {steps} 步后的画面质量衰减(漂移)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <path d={pathOff} fill="none" stroke="#9ca3af" strokeWidth={2} strokeDasharray={withAug ? "3 3" : undefined} opacity={withAug ? 0.4 : 1} />
        <path d={pathOn} fill="none" stroke="#d946ef" strokeWidth={2} strokeDasharray={withAug ? undefined : "3 3"} opacity={withAug ? 1 : 0.4} />
        <text x={W - 45} y={toY(curveOff[curveOff.length - 1]) + 4} textAnchor="end" fontSize={10} fill="#9ca3af">无增强</text>
        <text x={W - 45} y={toY(curveOn[curveOn.length - 1]) - 6} textAnchor="end" fontSize={10} fill="#d946ef">有增强</text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        不加噪声增强:模型训练时只见过"完美"的条件帧,推理时自己生成的略有瑕疵的帧作为下一步条件会导致误差越滚越大(灰线快速下滑)。加噪声增强后模型学会了容忍不完美条件帧,漂移显著减缓(粉线)。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/NoiseAugmentationStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAMENGEN_SOURCE_PATH } from "../lib/prose";
import { DriftCompareWidget } from "../widgets/DriftCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function NoiseAugmentationStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:噪声增强条件帧
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        训练时故意在条件帧上加噪声,让模型学会容忍自己此前生成的不完美画面 —— 这是对抗长程自回归漂移的核心工程细节。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <DriftCompareWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageGameNGen.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageGameNGen.tsx`**

```tsx
import { Link } from "react-router";
import gamengenMarkdown from "../../../../../../16-world-models/06-gamengen.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GAMENGEN_SOURCE_PATH } from "./lib/prose";
import { RlDataStage } from "./stages/RlDataStage";
import { DiffusionPredictStage } from "./stages/DiffusionPredictStage";
import { NoiseAugmentationStage } from "./stages/NoiseAugmentationStage";
import styles from "./NodePageGameNGen.module.css";

const prose = extractProse(gamengenMarkdown);

export default function NodePageGameNGen() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>GameNGen (2024)</h1>
        <div className={styles.metaLine}>作者:Dani Valevski · Yaniv Leviathan · Moab Arar · Shlomi Fruchter</div>
        <div className={styles.metaLine}>论文:Diffusion Models Are Real-Time Game Engines</div>
        <p className={styles.keyIdea}>
          用条件 diffusion 模型完全替代传统游戏引擎的渲染循环,实时交互式生成可玩的 DOOM 画面
        </p>
      </section>

      <section className={styles.stage}>
        <RlDataStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DiffusionPredictStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <NoiseAugmentationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GAMENGEN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GAMENGEN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "16-world-models/06-gamengen": lazy(() => import("./gamengen/NodePageGameNGen")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/gamengen web/src/components/node/golden/index.ts
git commit -m "feat: GameNGen 金标本 Stage3 + NodePage + 注册"
```

---

## Task 13: 全项目验证

**Files:** 无新建/修改,纯验证任务。

- [ ] **Step 1: 跑全量 vitest**

```bash
cd web && npx vitest run
```

Expected: 全部通过,`AllGoldenSamples.smoke.test.tsx` 新增 6 条(World Models/Video Diffusion Models/DreamerV3/Sora/Genie/GameNGen)冒烟用例 PASS,`ProseCompleteness.test.tsx` 新增 6 个 prose 模块用例 PASS,总用例数比 Task 开始前(327)多至少 12 条。

- [ ] **Step 2: 跑 tsc**

```bash
cd web && npx tsc --noEmit
```

Expected: 无错误。

- [ ] **Step 3: 浏览器验证**

用 preview_start 起 web/ 的 dev server,依次打开并确认交互式 Stage 渲染(非纯 markdown 兜底)、核心交互有响应、console 无 error:

- `/families/16-world-models/01-world-models`
- `/families/16-world-models/02-video-diffusion-models`
- `/families/16-world-models/03-dreamerv3`
- `/families/16-world-models/04-sora`
- `/families/16-world-models/05-genie`
- `/families/16-world-models/06-gamengen`

- [ ] **Step 4: 提交(若浏览器验证发现小问题并修复)**

```bash
git add -A
git commit -m "fix: 世界模型金标本浏览器验证发现的问题修复"
```

(若验证全部通过、无需修复,跳过此步)

---

## Plan Self-Review 记录

- **Spec 覆盖**:spec 第 3 节列出的 6 个节点、每节点 3 个 Stage 的设计要点均对应到 Task 1-12 里的具体 Stage/Widget;spec 第 6 节验收标准对应 Task 13。
- **prose 模板一致性**:确认本家族用 mixtral 式双层模板(H2_KEYS + H3_KEYS 仅在 `_coreInsight` 状态下生效),已在"关键背景"里给出完整代码,6 个节点直接复用,只改 `XXX_SOURCE_PATH` 常量。
- **Placeholder 扫描**:所有步骤均为完整可运行代码,无 TBD/TODO。
- **类型一致性**:`ProseSections` 接口在"关键背景"通用模板中统一定义一次,6 个节点的 `lib/prose.ts` 都从这个模板复制,字段名在所有 Stage 组件的 props 里保持一致引用。
