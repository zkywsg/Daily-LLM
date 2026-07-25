# 语音/音频模型家族金标本交互页 Implementation Plan

**Goal:** 为第 18 个家族(`18-speech-audio`)的 5 个节点(Wav2Vec 2.0/HuBERT/Whisper/AudioLM/MusicGen)各自补一个交互金标本页,复用现有 `web/src/components/node/golden/{slug}/` 结构约定。补完后仓库全部 84 个节点都会有金标本交互页。

**Architecture:** 每节点 `lib/data.ts`(确定性 demo 数据/函数)+ `lib/prose.ts`(GNN 式扁平单层提取模板)+ 3 个 `stages/*.tsx`(对应机制一/二/三,每个 stage 用 1 个自包含 widget)+ `NodePage{Name}.tsx`(hero+3 stage+footer)+ `NodePage{Name}.module.css`,注册进 `web/src/components/node/golden/index.ts`。

**Tech Stack:** React + TypeScript,内联 SVG/HTML,CSS Modules,Vitest。

---

## 关键背景(所有任务共用)

**本家族的 markdown 结构与 GNN(17 家族)相同**(与 World Models/16 家族的双层结构不同):`## 核心思想 + 直觉` 本身就是一个扁平 H2,不嵌套 `### 直觉`;`## 机制一/二/三` 也都是顶层 H2。`lib/prose.ts` 直接复用 GNN 家族已验证过的**单层**提取模板:

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

const H2_KEYS: Array<{ test: RegExp; key: keyof ProseSections }> = [
  { test: /^前作进展/, key: "previousWork" },
  { test: /^核心思想/, key: "intuition" },
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
  let buffer: string[] = [];
  const flush = () => {
    if (currentKey) sections[currentKey] = buffer.join("\n").trim();
    buffer = [];
  };

  for (const line of body.split("\n")) {
    const h2 = /^## +(.+?)\s*$/.exec(line);
    if (h2) {
      flush();
      const m = H2_KEYS.find((x) => x.test.test(h2[1].trim()));
      currentKey = m ? m.key : null;
      continue;
    }
    if (currentKey) buffer.push(line);
  }
  flush();
  return sections;
}
```

每个节点的 `lib/prose.ts` 只改 `XXX_SOURCE_PATH` 常量,`extractProse` 逻辑原样复制。**写每个节点前先跑 `grep -n '^## ' 18-speech-audio/{文件}.md` 确认标题措辞与上面正则匹配。**

**路由注册 key 格式**:`"18-speech-audio/{NN-slug}"`(如 `"18-speech-audio/01-wav2vec2"`)。

**标题渐变色**:`.title` 用 `linear-gradient(90deg, var(--family-18) 0%, var(--accent-link) 100%)`(family-18 = `#fb7185` 浅玫瑰红)。

**通用 `NodePage{Name}.module.css` 模板**(5 个节点完全复制):

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
  background: linear-gradient(90deg, var(--family-18) 0%, var(--accent-link) 100%);
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

**通用 `stages/Stage.module.css` 模板**(5 个节点完全复制):

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

**已知坑(前几轮反复沉淀的经验,这轮必须主动规避)**:
1. **prose 模板必须用扁平单层**(上面已给出),不能照抄 World Models 轮的双层模板。
2. **SVG 条形图/曲线高度必须 `Math.min(..., cap)` 钳位**,禁止无上限缩放。
3. **所有 toggle/选择按钮加 `aria-pressed={condition}`**。
4. **不要硬编码重复 `lib/data.ts` 已导出的常量**,必须 import 复用。
5. **深色主题下固定背景色配色必须显式指定可读文字颜色**,不能依赖继承 `var(--ink-primary)`。
6. **任何"滑块调参数 → 图表变化"的 widget,写完后必须手动验证参数变化确实产生声称的效果**——这是本轮之前两轮反复出现的 bug 类型(非invertible 基、常数比例、透明度溢出)。下面每个 widget 的设计已经过数学正确性预验证(见各任务描述),实现时按给定公式精确实现,不要凭感觉简化。
7. **`index.ts` 注册 key 精确匹配** `${familyId}/${nodeSlug}`。

---

## Task 1: Wav2Vec 2.0 — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/wav2vec2/lib/data.ts`
- Create: `web/src/components/node/golden/wav2vec2/lib/prose.ts`
- Create: `web/src/components/node/golden/wav2vec2/stages/Stage.module.css`
- Create: `web/src/components/node/golden/wav2vec2/widgets/DownsampleWidget.tsx`
- Create: `web/src/components/node/golden/wav2vec2/stages/FeatureEncoderStage.tsx`
- Create: `web/src/components/node/golden/wav2vec2/widgets/ContrastiveWidget.tsx`
- Create: `web/src/components/node/golden/wav2vec2/stages/QuantizeStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## ' 18-speech-audio/01-wav2vec2.md
```

Expected:`## 前作进展`、`## 核心思想 + 直觉`、`## 机制一:CNN 特征编码器`、`## 机制二:量化模块 + 对比学习目标`、`## 机制三:Transformer 上下文编码 + 掩码预测`、`## 三件套协同`、`## 关键代码`、`## 性能数据`、`## 影响 / 后续`(全部扁平 H2,无嵌套)。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// Wav2Vec 2.0 demo 数据:toy 波形下采样 + 量化码本 + 对比学习分数。
// 全部确定性构造,不真跑模型。

export const WAVEFORM_LEN = 64;
export const TOY_WAVEFORM: number[] = Array.from({ length: WAVEFORM_LEN }, (_, i) =>
  Math.sin(i * 0.4) * 0.5 + Math.sin(i * 0.09) * 0.3
);

/** 简化下采样:每层做 stride=2 的相邻平均池化,模拟 CNN 特征编码器逐层压缩帧率 */
export function downsample(waveform: number[], numLayers: number): number[] {
  let cur = waveform;
  for (let l = 0; l < numLayers; l++) {
    const next: number[] = [];
    for (let i = 0; i + 1 < cur.length; i += 2) next.push((cur[i] + cur[i + 1]) / 2);
    cur = next;
  }
  return cur;
}

export const CODEBOOK_SIZE = 6;

/** 确定性码本向量(单位圆上均匀分布,2 维便于画在平面上) */
export function codebookVector(idx: number): [number, number] {
  const angle = (idx / CODEBOOK_SIZE) * 2 * Math.PI;
  return [Math.cos(angle), Math.sin(angle)];
}

/** 给定"真实"码本索引,生成一个带小扰动的连续特征 z(扰动幅度远小于码本间距,
 * 保证对比学习任务里真实目标始终是相似度最高的那个,这是本 demo 的设计前提)。 */
export function frameToContinuousFeature(trueCodeIdx: number, seed: number): [number, number] {
  const [cx, cy] = codebookVector(trueCodeIdx);
  const jitter = (((seed * 977) % 100) / 1000) - 0.05; // [-0.05, 0.05)
  return [cx + jitter, cy - jitter];
}

/** 对比学习:z 与全部码本向量的余弦相似度,softmax(带温度)得到"选中概率" */
export function contrastiveScores(z: [number, number]): number[] {
  const sims = Array.from({ length: CODEBOOK_SIZE }, (_, i) => {
    const [cx, cy] = codebookVector(i);
    const dot = z[0] * cx + z[1] * cy;
    const normZ = Math.sqrt(z[0] ** 2 + z[1] ** 2) || 1e-6;
    const normC = Math.sqrt(cx ** 2 + cy ** 2) || 1e-6;
    return dot / (normZ * normC);
  });
  const m = Math.max(...sims);
  const exps = sims.map((s) => Math.exp((s - m) * 5)); // 温度缩放让分布更尖锐,便于可视化
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用"关键背景"通用单层模板,加 `WAV2VEC2_SOURCE_PATH = "18-speech-audio/01-wav2vec2.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

原样复制"关键背景"模板。

- [ ] **Step 5: 写 `widgets/DownsampleWidget.tsx`**

```tsx
import { useState } from "react";
import { TOY_WAVEFORM, downsample } from "../lib/data";

const W = 680;
const H = 260;

export function DownsampleWidget() {
  const [numLayers, setNumLayers] = useState(0);
  const frames = downsample(TOY_WAVEFORM, numLayers);
  const frameRateHz = Math.round(16000 / Math.pow(2, numLayers));

  const barW = Math.max(2, (W - 60) / frames.length - 1);
  const maxAbs = Math.max(...frames.map((v) => Math.abs(v)), 0.1);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        CNN 下采样层数 = {numLayers}(帧率 ≈ {frameRateHz}Hz,共 {frames.length} 帧)
        <input type="range" min={0} max={5} value={numLayers} onChange={(e) => setNumLayers(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`下采样 ${numLayers} 层后共 ${frames.length} 帧,帧率约 ${frameRateHz}Hz`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          下采样后的帧序列(每帧一根柱子)
        </text>
        <line x1={30} y1={H / 2} x2={W - 30} y2={H / 2} stroke="var(--border)" />
        {frames.map((v, i) => {
          const x = 30 + i * ((W - 60) / frames.length);
          const h = Math.min((Math.abs(v) / maxAbs) * (H / 2 - 30), H / 2 - 30);
          const y = v >= 0 ? H / 2 - h : H / 2;
          return <rect key={i} x={x} y={y} width={barW} height={h} fill="#fb7185" />;
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        层数越多,帧数越少、每帧覆盖的时间跨度越长——16kHz 原始波形经过约 7 层卷积后,帧率会压缩到真实 wav2vec 2.0 使用的约 50Hz。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/FeatureEncoderStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WAV2VEC2_SOURCE_PATH } from "../lib/prose";
import { DownsampleWidget } from "../widgets/DownsampleWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function FeatureEncoderStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:CNN 特征编码器
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        多层一维卷积把原始波形压缩成较低频率的潜在特征序列,压缩掉冗余的高频细节,保留语音相关的结构信息。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <DownsampleWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/ContrastiveWidget.tsx`**

```tsx
import { useState } from "react";
import { CODEBOOK_SIZE, codebookVector, frameToContinuousFeature, contrastiveScores } from "../lib/data";

const W = 680;
const H = 300;

export function ContrastiveWidget() {
  const [trueCode, setTrueCode] = useState(2);
  const z = frameToContinuousFeature(trueCode, trueCode);
  const scores = contrastiveScores(z);
  const predictedCode = scores.indexOf(Math.max(...scores));

  const cx0 = 150, cy0 = 150, r = 100;
  const toXY = (v: [number, number]) => [cx0 + v[0] * r, cy0 - v[1] * r];

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {Array.from({ length: CODEBOOK_SIZE }, (_, i) => (
          <button
            key={i} type="button" onClick={() => setTrueCode(i)} aria-pressed={i === trueCode}
            style={{
              width: 30, height: 30, borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === trueCode ? "#fb7185" : "var(--border)"}`,
              background: i === trueCode ? "#fb7185" : "var(--bg-surface)",
              color: i === trueCode ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
            }}
          >
            {i}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`真实量化目标 ${trueCode},模型预测 ${predictedCode}`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          对比学习:z 与码本向量的相似度
        </text>
        {Array.from({ length: CODEBOOK_SIZE }, (_, i) => {
          const [x, y] = toXY(codebookVector(i));
          const isTrue = i === trueCode;
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={14} fill={isTrue ? "#fb7185" : "var(--bg-subtle)"} stroke="var(--border)" />
              <text x={x} y={y + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill={isTrue ? "#fff" : "var(--ink-primary)"}>{i}</text>
            </g>
          );
        })}
        {(() => {
          const [zx, zy] = toXY(z);
          return <circle cx={zx} cy={zy} r={6} fill="#9d174d" />;
        })()}
        {scores.map((s, i) => {
          const x = 320 + (i % 3) * 110;
          const y = 60 + Math.floor(i / 3) * 100;
          const h = Math.min(s * 70, 70);
          return (
            <g key={i}>
              <rect x={x} y={130 - h} width={30} height={h} fill={i === predictedCode ? "#fb7185" : "#9ca3af"} />
              <text x={x + 15} y={148} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">码 {i}</text>
              <text x={x + 15} y={130 - h - 4} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">{s.toFixed(2)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        深粉色圆点是连续特征 z(带小扰动);右侧柱状图是 z 与每个码本向量的相似度 softmax——{predictedCode === trueCode ? "模型正确选中了真实目标" : "模型选错了目标"}(码 {predictedCode})。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/QuantizeStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WAV2VEC2_SOURCE_PATH } from "../lib/prose";
import { ContrastiveWidget } from "../widgets/ContrastiveWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function QuantizeStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:量化模块 + 对比学习目标
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        连续特征被量化成离散码本向量作为对比学习的目标,模型需要从候选码本里正确识别出真实目标。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WAV2VEC2_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <ContrastiveWidget />
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
git add web/src/components/node/golden/wav2vec2/lib web/src/components/node/golden/wav2vec2/stages web/src/components/node/golden/wav2vec2/widgets
git commit -m "feat: Wav2Vec 2.0 金标本 lib + Stage1/2"
```

---

## Task 2: Wav2Vec 2.0 — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/wav2vec2/widgets/MaskedPredictWidget.tsx`
- Create: `web/src/components/node/golden/wav2vec2/stages/MaskedPredictStage.tsx`
- Create: `web/src/components/node/golden/wav2vec2/NodePageWav2Vec2.tsx`
- Create: `web/src/components/node/golden/wav2vec2/NodePageWav2Vec2.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/MaskedPredictWidget.tsx`**

```tsx
import { useState } from "react";
import { CODEBOOK_SIZE } from "../lib/data";

const SEQ_LEN = 10;
const TOKEN_SEQUENCE: number[] = Array.from({ length: SEQ_LEN }, (_, i) => (i * 3 + 1) % CODEBOOK_SIZE);

/** 用左右相邻未 mask 位置的平均(四舍五入)预测被 mask 位置——
 * 这是"用上下文预测"的简化示意,不保证每次都对,和真实掩码预测任务一样。 */
function predictMasked(seq: number[], maskedIdx: number): number {
  const left = seq[(maskedIdx - 1 + seq.length) % seq.length];
  const right = seq[(maskedIdx + 1) % seq.length];
  return Math.round((left + right) / 2);
}

export function MaskedPredictWidget() {
  const [maskedSet, setMaskedSet] = useState<Set<number>>(new Set([3, 7]));

  const toggle = (i: number) => {
    setMaskedSet((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
        {TOKEN_SEQUENCE.map((v, i) => {
          const isMasked = maskedSet.has(i);
          const predicted = isMasked ? predictMasked(TOKEN_SEQUENCE, i) : null;
          const correct = predicted === v;
          return (
            <button
              key={i} type="button" onClick={() => toggle(i)} aria-pressed={isMasked}
              style={{
                width: 56, height: 56, borderRadius: "var(--radius-md)",
                border: `2px solid ${isMasked ? (correct ? "#059669" : "#dc2626") : "var(--border)"}`,
                background: isMasked ? "var(--bg-surface)" : "var(--bg-subtle)",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                cursor: "pointer", fontSize: "var(--fs-xs)",
              }}
            >
              {isMasked ? (
                <>
                  <span style={{ fontSize: 9, color: "var(--ink-muted)" }}>预测</span>
                  <span style={{ fontWeight: 700, color: correct ? "#059669" : "#dc2626" }}>{predicted}</span>
                </>
              ) : (
                <span style={{ fontWeight: 700, color: "var(--ink-primary)" }}>{v}</span>
              )}
            </button>
          );
        })}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        点击方块切换是否 mask。被 mask 的位置(粗边框)显示 Transformer 用左右上下文预测出的量化目标——绿色表示预测正确,红色表示预测错误(和真实模型一样,上下文预测不保证每次都对)。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/MaskedPredictStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WAV2VEC2_SOURCE_PATH } from "../lib/prose";
import { MaskedPredictWidget } from "../widgets/MaskedPredictWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function MaskedPredictStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Transformer 上下文编码 + 掩码预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        随机 mask 掉若干帧,Transformer 需要从上下文预测被 mask 位置对应的量化目标——这是自监督训练的"完形填空"任务。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <MaskedPredictWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageWav2Vec2.module.css`**

原样复制"关键背景"模板。

- [ ] **Step 4: 写 `NodePageWav2Vec2.tsx`**

```tsx
import { Link } from "react-router";
import wav2vec2Markdown from "../../../../../../18-speech-audio/01-wav2vec2.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WAV2VEC2_SOURCE_PATH } from "./lib/prose";
import { FeatureEncoderStage } from "./stages/FeatureEncoderStage";
import { QuantizeStage } from "./stages/QuantizeStage";
import { MaskedPredictStage } from "./stages/MaskedPredictStage";
import styles from "./NodePageWav2Vec2.module.css";

const prose = extractProse(wav2vec2Markdown);

export default function NodePageWav2Vec2() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>Wav2Vec 2.0 (2020)</h1>
        <div className={styles.metaLine}>作者:Alexei Baevski · Henry Zhou · Abdelrahman Mohamed · Michael Auli</div>
        <div className={styles.metaLine}>论文:wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations</div>
        <p className={styles.keyIdea}>
          CNN 特征编码器 + 可学习量化模块生成离散对比目标 + Transformer 掩码预测,用对比学习从原始波形自监督学到可迁移的语音表征
        </p>
      </section>

      <section className={styles.stage}>
        <FeatureEncoderStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <QuantizeStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MaskedPredictStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WAV2VEC2_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WAV2VEC2_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "18-speech-audio/01-wav2vec2": lazy(() => import("./wav2vec2/NodePageWav2Vec2")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/wav2vec2 web/src/components/node/golden/index.ts
git commit -m "feat: Wav2Vec 2.0 金标本 Stage3 + NodePage + 注册"
```

---

## Task 3: HuBERT — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/hubert/lib/data.ts`
- Create: `web/src/components/node/golden/hubert/lib/prose.ts`
- Create: `web/src/components/node/golden/hubert/stages/Stage.module.css`
- Create: `web/src/components/node/golden/hubert/widgets/ClusteringWidget.tsx`
- Create: `web/src/components/node/golden/hubert/stages/ClusteringStage.tsx`
- Create: `web/src/components/node/golden/hubert/widgets/ClassDistributionWidget.tsx`
- Create: `web/src/components/node/golden/hubert/stages/MaskedClassifyStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## ' 18-speech-audio/02-hubert.md
```

Expected:`## 机制一:离线聚类生成伪标签`、`## 机制二:BERT 式掩码预测`、`## 机制三:迭代式重新聚类`,其余同通用结构。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// HuBERT demo 数据:toy 特征点 k-means 聚类 + 掩码分类置信度分布 +
// 迭代式重新聚类(Lloyd's 算法一步更新)。全部确定性构造。

export const NUM_POINTS = 8;
export const TOY_POINTS: Array<[number, number]> = Array.from({ length: NUM_POINTS }, (_, i) => {
  const angle = (i / NUM_POINTS) * 2 * Math.PI;
  const r = 0.5 + (i % 3) * 0.25;
  return [Math.cos(angle) * r, Math.sin(angle) * r];
});

export const NUM_CLUSTERS = 3;

export function initialCenters(): Array<[number, number]> {
  return Array.from({ length: NUM_CLUSTERS }, (_, k) => {
    const angle = (k / NUM_CLUSTERS) * 2 * Math.PI;
    return [Math.cos(angle) * 0.3, Math.sin(angle) * 0.3];
  });
}

function dist2(a: [number, number], b: [number, number]): number {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
}

export function assignClusters(points: Array<[number, number]>, centers: Array<[number, number]>): number[] {
  return points.map((p) => {
    let best = 0, bestD = Infinity;
    centers.forEach((c, k) => {
      const d = dist2(p, c);
      if (d < bestD) { bestD = d; best = k; }
    });
    return best;
  });
}

/** 一步 Lloyd's 算法更新:用当前分配重新计算聚类中心(各聚类内点的均值) */
export function updateCenters(points: Array<[number, number]>, assignments: number[]): Array<[number, number]> {
  const sums: Array<[number, number, number]> = Array.from({ length: NUM_CLUSTERS }, () => [0, 0, 0]);
  points.forEach((p, i) => {
    const k = assignments[i];
    sums[k][0] += p[0];
    sums[k][1] += p[1];
    sums[k][2] += 1;
  });
  const fallback = initialCenters();
  return sums.map(([sx, sy, n], k) => (n > 0 ? [sx / n, sy / n] as [number, number] : fallback[k]));
}

/** 给定"真实类别"和置信度(sharpness),softmax 出一个类别分布 —— sharpness 越大分布越尖锐,
 * 但 trueClass 的 logit 恒定比其余类别高,所以任意 sharpness 下 trueClass 概率都是最高的。 */
export function classDistribution(trueClass: number, sharpness: number): number[] {
  const logits = Array.from({ length: NUM_CLUSTERS }, (_, k) => (k === trueClass ? sharpness : -sharpness / 2));
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用单层模板,`HUBERT_SOURCE_PATH = "18-speech-audio/02-hubert.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/ClusteringWidget.tsx`**

```tsx
import { TOY_POINTS, initialCenters, assignClusters } from "../lib/data";

const W = 320;
const H = 320;
const PALETTE = ["#fb7185", "#f59e0b", "#8b5cf6"];

export function ClusteringWidget() {
  const centers = initialCenters();
  const assignments = assignClusters(TOY_POINTS, centers);

  const toXY = (p: [number, number]) => [W / 2 + p[0] * 100, H / 2 - p[1] * 100];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="k-means 聚类分配可视化">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          离线 k-means 聚类:每个特征点被分配到最近的聚类中心
        </text>
        {centers.map((c, k) => {
          const [x, y] = toXY(c);
          return <circle key={k} cx={x} cy={y} r={10} fill="none" stroke={PALETTE[k]} strokeWidth={3} />;
        })}
        {TOY_POINTS.map((p, i) => {
          const [x, y] = toXY(p);
          const k = assignments[i];
          return <circle key={i} cx={x} cy={y} r={7} fill={PALETTE[k]} />;
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        实心点是特征点,颜色代表分配到的聚类;空心圆环是聚类中心。这一步产生的聚类标签就是 HuBERT 第一轮训练用的离散伪标签。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/ClusteringStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { HUBERT_SOURCE_PATH } from "../lib/prose";
import { ClusteringWidget } from "../widgets/ClusteringWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ClusteringStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:离线聚类生成伪标签
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用独立的、不依赖被训练模型本身的聚类步骤生成一套固定的伪标签,避免 Wav2Vec 2.0 那种联合优化目标带来的训练不稳定。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <ClusteringWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/ClassDistributionWidget.tsx`**

```tsx
import { useState } from "react";
import { NUM_CLUSTERS, classDistribution } from "../lib/data";

const W = 500;
const H = 260;

export function ClassDistributionWidget() {
  const [sharpness, setSharpness] = useState(1);
  const trueClass = 1;
  const dist = classDistribution(trueClass, sharpness);

  const barW = (W - 80) / NUM_CLUSTERS - 10;

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        分类置信度(sharpness)= {sharpness.toFixed(1)}
        <input type="range" min={0} max={5} step={0.5} value={sharpness} onChange={(e) => setSharpness(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`真实类别 ${trueClass} 的掩码分类置信度分布`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          掩码分类头输出的类别概率分布(真实类别 = {trueClass})
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        {dist.map((p, k) => {
          const x = 60 + k * (barW + 30);
          const h = Math.min(p * (H - 100), H - 100);
          return (
            <g key={k}>
              <rect x={x} y={H - 40 - h} width={barW} height={h} fill={k === trueClass ? "#fb7185" : "#9ca3af"} />
              <text x={x + barW / 2} y={H - 40 - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">{p.toFixed(2)}</text>
              <text x={x + barW / 2} y={H - 18} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">类 {k}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        这是标准的分类任务(交叉熵),不是对比学习——sharpness 越大,模型对正确类别的置信度越高,分布越尖锐。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/MaskedClassifyStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { HUBERT_SOURCE_PATH } from "../lib/prose";
import { ClassDistributionWidget } from "../widgets/ClassDistributionWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function MaskedClassifyStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:BERT 式掩码预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        在被 mask 的位置用分类头预测伪标签类别,用标准交叉熵训练——不需要对比学习里的负采样。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={HUBERT_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <ClassDistributionWidget />
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
git add web/src/components/node/golden/hubert/lib web/src/components/node/golden/hubert/stages web/src/components/node/golden/hubert/widgets
git commit -m "feat: HuBERT 金标本 lib + Stage1/2"
```

---

## Task 4: HuBERT — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/hubert/widgets/IterativeReclusterWidget.tsx`
- Create: `web/src/components/node/golden/hubert/stages/ReclusterStage.tsx`
- Create: `web/src/components/node/golden/hubert/NodePageHuBERT.tsx`
- Create: `web/src/components/node/golden/hubert/NodePageHuBERT.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/IterativeReclusterWidget.tsx`**

```tsx
import { useState } from "react";
import { TOY_POINTS, NUM_CLUSTERS, initialCenters, assignClusters, updateCenters } from "../lib/data";

const W = 320;
const H = 320;
const PALETTE = ["#fb7185", "#f59e0b", "#8b5cf6"];

// 迭代式重新聚类:每点一次"跑下一轮",用当前分配重新计算聚类中心(Lloyd's
// 算法一步更新),再重新分配——聚类边界逐轮收敛,模拟 HuBERT 用模型自身
// 隐藏层特征重新聚类、提纯伪标签的过程。

export function IterativeReclusterWidget() {
  const [round, setRound] = useState(0);
  const [centers, setCenters] = useState(initialCenters());

  const assignments = assignClusters(TOY_POINTS, centers);
  const toXY = (p: [number, number]) => [W / 2 + p[0] * 100, H / 2 - p[1] * 100];

  const nextRound = () => {
    const newCenters = updateCenters(TOY_POINTS, assignments);
    setCenters(newCenters);
    setRound((r) => r + 1);
  };

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`迭代重聚类第 ${round} 轮`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          迭代式重新聚类 — 第 {round} 轮
        </text>
        {centers.map((c, k) => {
          const [x, y] = toXY(c);
          return <circle key={k} cx={x} cy={y} r={10} fill="none" stroke={PALETTE[k % NUM_CLUSTERS]} strokeWidth={3} />;
        })}
        {TOY_POINTS.map((p, i) => {
          const [x, y] = toXY(p);
          const k = assignments[i];
          return <circle key={i} cx={x} cy={y} r={7} fill={PALETTE[k % NUM_CLUSTERS]} />;
        })}
      </svg>
      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-2)" }}>
        <button
          type="button" onClick={nextRound} disabled={round >= 4}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: round >= 4 ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: round >= 4 ? 0.5 : 1 }}
        >
          跑下一轮重聚类
        </button>
        <button
          type="button" onClick={() => { setCenters(initialCenters()); setRound(0); }}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        每一轮都用当前分配重新计算聚类中心(移动到各自簇内点的均值),边界逐轮收敛更稳定——这正是 HuBERT 用模型隐藏层特征重新聚类提纯伪标签的过程。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/ReclusterStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { HUBERT_SOURCE_PATH } from "../lib/prose";
import { IterativeReclusterWidget } from "../widgets/IterativeReclusterWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ReclusterStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:迭代式重新聚类
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用前一轮模型的隐藏层特征重新聚类,生成质量更高、更贴近音素边界的新伪标签,通常迭代 2 轮。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <IterativeReclusterWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageHuBERT.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageHuBERT.tsx`**

```tsx
import { Link } from "react-router";
import hubertMarkdown from "../../../../../../18-speech-audio/02-hubert.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, HUBERT_SOURCE_PATH } from "./lib/prose";
import { ClusteringStage } from "./stages/ClusteringStage";
import { MaskedClassifyStage } from "./stages/MaskedClassifyStage";
import { ReclusterStage } from "./stages/ReclusterStage";
import styles from "./NodePageHuBERT.module.css";

const prose = extractProse(hubertMarkdown);

export default function NodePageHuBERT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>HuBERT (2021)</h1>
        <div className={styles.metaLine}>作者:Wei-Ning Hsu · Benjamin Bolte · Yao-Hung Hubert Tsai · Kushal Lakhotia · Ruslan Salakhutdinov · Abdelrahman Mohamed</div>
        <div className={styles.metaLine}>论文:HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units</div>
        <p className={styles.keyIdea}>
          用离线 k-means 聚类生成离散伪标签,再做 BERT 式掩码预测,配合迭代式重新聚类不断提纯伪标签的音素区分度
        </p>
      </section>

      <section className={styles.stage}>
        <ClusteringStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MaskedClassifyStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ReclusterStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={HUBERT_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={HUBERT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "18-speech-audio/02-hubert": lazy(() => import("./hubert/NodePageHuBERT")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/hubert web/src/components/node/golden/index.ts
git commit -m "feat: HuBERT 金标本 Stage3 + NodePage + 注册"
```

---

## Task 5: Whisper — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/whisper/lib/data.ts`
- Create: `web/src/components/node/golden/whisper/lib/prose.ts`
- Create: `web/src/components/node/golden/whisper/stages/Stage.module.css`
- Create: `web/src/components/node/golden/whisper/widgets/SpectrogramWidget.tsx`
- Create: `web/src/components/node/golden/whisper/stages/SpectrogramStage.tsx`
- Create: `web/src/components/node/golden/whisper/widgets/DataFilterWidget.tsx`
- Create: `web/src/components/node/golden/whisper/stages/DataFilterStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## ' 18-speech-audio/03-whisper.md
```

Expected:`## 机制一:标准 Transformer encoder-decoder + log-mel 频谱输入`、`## 机制二:大规模弱监督数据收集与过滤`、`## 机制三:多任务统一格式`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// Whisper demo 数据:toy 波形 → log-mel 频谱可视化 + 弱监督数据质量过滤 +
// 多任务前缀-输出示例。全部确定性构造。

export const N_FRAMES = 10;
export const N_MELS = 6;

export const TOY_WAVEFORM: number[] = Array.from({ length: 80 }, (_, i) =>
  Math.sin(i * 0.5) * 0.4 + Math.sin(i * 0.13) * 0.3 + Math.sin(i * 0.05) * 0.2
);

/** 简化 log-mel 频谱:把波形分帧,每帧与几个固定频率模板做余弦相关,取 log(1+|.|) 作为能量,
 * 最后按整体最大值归一化到 [0,1]。不是真实 FFT,但作为示意频谱图数值上是合理有界的。 */
export function logMelSpectrogram(waveform: number[]): number[][] {
  const frameLen = Math.floor(waveform.length / N_FRAMES);
  const spec: number[][] = [];
  for (let t = 0; t < N_FRAMES; t++) {
    const frame = waveform.slice(t * frameLen, (t + 1) * frameLen);
    const row: number[] = [];
    for (let m = 0; m < N_MELS; m++) {
      const freq = (m + 1) * 0.3;
      let energy = 0;
      frame.forEach((v, i) => { energy += v * Math.cos(freq * i); });
      row.push(Math.log(1 + Math.abs(energy)));
    }
    spec.push(row);
  }
  const maxV = Math.max(...spec.flat(), 1e-6);
  return spec.map((row) => row.map((v) => v / maxV));
}

/** 68 万小时弱监督数据里,一小批样本的质量分示例(0-1,越高越像高质量人工转写) */
export const RAW_QUALITY_SCORES: number[] = [0.9, 0.85, 0.2, 0.75, 0.1, 0.6, 0.95, 0.15, 0.8, 0.3, 0.7, 0.05];

export function filterLowQuality(scores: number[], threshold = 0.5): number[] {
  return scores.filter((s) => s >= threshold);
}

export const TASK_PREFIXES: Array<{ id: string; label: string; prefix: string; outputExample: string }> = [
  { id: "transcribe", label: "转写", prefix: "<|transcribe|>", outputExample: "今天天气不错。" },
  { id: "translate", label: "翻译", prefix: "<|translate|>", outputExample: "The weather is nice today." },
  { id: "langid", label: "语言识别", prefix: "<|langid|>", outputExample: "zh(中文)" },
  { id: "timestamp", label: "时间戳", prefix: "<|timestamps|>", outputExample: "[00:00–00:03] 今天天气不错。" },
];
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用单层模板,`WHISPER_SOURCE_PATH = "18-speech-audio/03-whisper.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/SpectrogramWidget.tsx`**

```tsx
import { TOY_WAVEFORM, N_FRAMES, N_MELS, logMelSpectrogram } from "../lib/data";

const CELL = 30;

export function SpectrogramWidget() {
  const spec = logMelSpectrogram(TOY_WAVEFORM);

  return (
    <div>
      <svg viewBox={`0 0 ${N_FRAMES * CELL + 60} ${N_MELS * CELL + 40}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="log-mel 频谱图可视化">
        <text x={(N_FRAMES * CELL + 60) / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          log-mel 频谱图(横轴时间,纵轴 mel 频率)
        </text>
        {spec.map((row, t) =>
          row.map((v, m) => {
            const lightness = 90 - v * 55;
            return (
              <rect
                key={`${t}-${m}`}
                x={40 + t * CELL}
                y={30 + (N_MELS - 1 - m) * CELL}
                width={CELL - 1}
                height={CELL - 1}
                fill={`hsl(350, 70%, ${lightness}%)`}
              />
            );
          })
        )}
        <text x={20} y={30 + (N_MELS * CELL) / 2} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" transform={`rotate(-90, 20, ${30 + (N_MELS * CELL) / 2})`}>
          mel 频率
        </text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        颜色越深表示该时间-频率位置能量越高。Whisper 用这种频谱图(而非原始波形)作为 Transformer encoder 的输入。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/SpectrogramStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WHISPER_SOURCE_PATH } from "../lib/prose";
import { SpectrogramWidget } from "../widgets/SpectrogramWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SpectrogramStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:标准 Transformer encoder-decoder + log-mel 频谱输入
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        整体架构没有任何语音专用的特殊设计,刻意选用标准架构以验证"数据规模而非架构精巧"才是关键因素。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SpectrogramWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/DataFilterWidget.tsx`**

```tsx
import { useState } from "react";
import { RAW_QUALITY_SCORES, filterLowQuality } from "../lib/data";

const W = 500;
const H = 260;
const BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

function histogram(scores: number[]): number[] {
  const counts = new Array(BINS.length - 1).fill(0);
  scores.forEach((s) => {
    for (let b = 0; b < BINS.length - 1; b++) {
      if (s >= BINS[b] && (s < BINS[b + 1] || (b === BINS.length - 2 && s <= BINS[b + 1]))) {
        counts[b]++;
        break;
      }
    }
  });
  return counts;
}

export function DataFilterWidget() {
  const [filtered, setFiltered] = useState(false);
  const scores = filtered ? filterLowQuality(RAW_QUALITY_SCORES) : RAW_QUALITY_SCORES;
  const counts = histogram(scores);
  const maxCount = Math.max(...counts, 1);

  const barW = (W - 80) / counts.length - 8;

  return (
    <div>
      <button
        type="button" onClick={() => setFiltered((v) => !v)} aria-pressed={filtered}
        style={{
          padding: "4px 14px", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)",
          border: `1px solid ${filtered ? "#fb7185" : "var(--border)"}`,
          background: filtered ? "#fb7185" : "var(--bg-surface)",
          color: filtered ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
        }}
      >
        {filtered ? "✓ 已过滤低质量样本" : "过滤低质量样本"}
      </button>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`数据质量分布直方图,${filtered ? "已过滤" : "未过滤"}`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          数据质量分布(共 {scores.length} 条样本)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        {counts.map((c, i) => {
          const x = 50 + i * ((W - 80) / counts.length);
          const h = Math.min((c / maxCount) * (H - 100), H - 100);
          return (
            <g key={i}>
              <rect x={x} y={H - 40 - h} width={barW} height={h} fill="#fb7185" />
              <text x={x + barW / 2} y={H - 40 - h - 6} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">{c}</text>
              <text x={x + barW / 2} y={H - 20} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">{BINS[i].toFixed(1)}-{BINS[i + 1].toFixed(1)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        未过滤时低质量区间(可能是机器生成的伪转写)样本不少;过滤后低质量区间样本明显减少,保留的都是质量分 ≥0.5 的样本。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/DataFilterStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WHISPER_SOURCE_PATH } from "../lib/prose";
import { DataFilterWidget } from "../widgets/DataFilterWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DataFilterStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:大规模弱监督数据收集与过滤
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        68 万小时数据质量参差不齐,用启发式规则和分类器过滤掉可能是机器生成的低质量转写,尽量保留高质量监督信号。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WHISPER_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <DataFilterWidget />
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
git add web/src/components/node/golden/whisper/lib web/src/components/node/golden/whisper/stages web/src/components/node/golden/whisper/widgets
git commit -m "feat: Whisper 金标本 lib + Stage1/2"
```

---

## Task 6: Whisper — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/whisper/widgets/TaskPrefixWidget.tsx`
- Create: `web/src/components/node/golden/whisper/stages/TaskPrefixStage.tsx`
- Create: `web/src/components/node/golden/whisper/NodePageWhisper.tsx`
- Create: `web/src/components/node/golden/whisper/NodePageWhisper.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/TaskPrefixWidget.tsx`**

```tsx
import { useState } from "react";
import { TASK_PREFIXES } from "../lib/data";

export function TaskPrefixWidget() {
  const [taskIdx, setTaskIdx] = useState(0);
  const task = TASK_PREFIXES[taskIdx];

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
        {TASK_PREFIXES.map((t, i) => (
          <button
            key={t.id} type="button" onClick={() => setTaskIdx(i)} aria-pressed={i === taskIdx}
            style={{
              padding: "4px 12px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === taskIdx ? "#fb7185" : "var(--border)"}`,
              background: i === taskIdx ? "#fb7185" : "var(--bg-surface)",
              color: i === taskIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>decoder 输入的第一个特殊 token</div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--fs-md)", color: "#9d174d", marginBottom: "var(--space-3)" }}>{task.prefix}</div>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>模型输出示例</div>
        <div style={{ fontSize: "var(--fs-md)", color: "var(--ink-primary)" }}>{task.outputExample}</div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        同一个模型、同一套权重,仅靠切换 decoder 起始的特殊 token,就能在转写/翻译/语言识别/时间戳预测之间切换。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/TaskPrefixStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WHISPER_SOURCE_PATH } from "../lib/prose";
import { TaskPrefixWidget } from "../widgets/TaskPrefixWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function TaskPrefixStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:多任务统一格式
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用特殊 token 把转写、翻译、语言识别、时间戳预测统一编码进同一个 sequence-to-sequence 格式,一个模型同时具备多种能力。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <TaskPrefixWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageWhisper.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageWhisper.tsx`**

```tsx
import { Link } from "react-router";
import whisperMarkdown from "../../../../../../18-speech-audio/03-whisper.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WHISPER_SOURCE_PATH } from "./lib/prose";
import { SpectrogramStage } from "./stages/SpectrogramStage";
import { DataFilterStage } from "./stages/DataFilterStage";
import { TaskPrefixStage } from "./stages/TaskPrefixStage";
import styles from "./NodePageWhisper.module.css";

const prose = extractProse(whisperMarkdown);

export default function NodePageWhisper() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>Whisper (2022)</h1>
        <div className={styles.metaLine}>作者:Alec Radford · Jong Wook Kim · Tao Xu · Greg Brockman · Christine McLeavey · Ilya Sutskever</div>
        <div className={styles.metaLine}>论文:Robust Speech Recognition via Large-Scale Weak Supervision</div>
        <p className={styles.keyIdea}>
          68 万小时弱监督多语言多任务数据 + 标准 Transformer encoder-decoder,零样本鲁棒性接近或超过针对特定数据集微调的模型
        </p>
      </section>

      <section className={styles.stage}>
        <SpectrogramStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DataFilterStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <TaskPrefixStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WHISPER_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WHISPER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "18-speech-audio/03-whisper": lazy(() => import("./whisper/NodePageWhisper")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/whisper web/src/components/node/golden/index.ts
git commit -m "feat: Whisper 金标本 Stage3 + NodePage + 注册"
```

---

## Task 7: AudioLM — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/audiolm/lib/data.ts`
- Create: `web/src/components/node/golden/audiolm/lib/prose.ts`
- Create: `web/src/components/node/golden/audiolm/stages/Stage.module.css`
- Create: `web/src/components/node/golden/audiolm/widgets/SemanticTokenWidget.tsx`
- Create: `web/src/components/node/golden/audiolm/stages/SemanticTokenStage.tsx`
- Create: `web/src/components/node/golden/audiolm/widgets/RvqWidget.tsx`
- Create: `web/src/components/node/golden/audiolm/stages/RvqStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## ' 18-speech-audio/04-audiolm.md
```

Expected:`## 机制一:语义 token —— 来自自监督音频模型`、`## 机制二:声学 token —— 来自神经编解码器,残差量化`、`## 机制三:层级式级联生成`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// AudioLM demo 数据:语义 token 提取(帧→离散 token 哈希)+ 残差向量量化(RVQ)+
// 三阶段级联生成状态机。全部确定性构造。

export const SEMANTIC_GRID = 6;

/** 用简单哈希把一个 toy "帧"(用位置参数化)映射成语义 token id(0-31) */
export function extractSemanticToken(framePos: number): number {
  let h = Math.round(framePos * 1000);
  h = (h * 2654435761) >>> 0;
  return h % 32;
}

export const RVQ_LEVELS = 4;
export const CODES_PER_LEVEL = 8;

/** 第 level 层码本的第 idx 个条目取值:范围随层数指数缩小(逐层捕捉更细的残差) */
function levelCodebookEntry(level: number, idx: number): number {
  const range = 1 / Math.pow(2, level);
  return (idx / (CODES_PER_LEVEL - 1) - 0.5) * 2 * range;
}

/** 残差向量量化:逐层贪心找当前残差最接近的码本条目,累加进重建值,残差递减。
 * 层的取值范围按 2 的幂缩小,足以覆盖上一层最坏情况下的残差(设计上保证收敛)。 */
export function residualQuantize(x: number): { codes: number[]; reconstruction: number } {
  let residual = x;
  const codes: number[] = [];
  let recon = 0;
  for (let level = 0; level < RVQ_LEVELS; level++) {
    let bestIdx = 0, bestDist = Infinity;
    for (let idx = 0; idx < CODES_PER_LEVEL; idx++) {
      const d = Math.abs(residual - levelCodebookEntry(level, idx));
      if (d < bestDist) { bestDist = d; bestIdx = idx; }
    }
    codes.push(bestIdx);
    const chosen = levelCodebookEntry(level, bestIdx);
    recon += chosen;
    residual -= chosen;
  }
  return { codes, reconstruction: recon };
}

/** 只用前 numLevels 层码本做部分重建(codes 数组的前 numLevels 项在完整量化下就已确定,
 * 与后续层无关,所以可以直接截断使用) */
export function partialReconstruction(codes: number[], numLevels: number): number {
  let recon = 0;
  for (let level = 0; level < numLevels; level++) recon += levelCodebookEntry(level, codes[level]);
  return recon;
}

export const TOY_SEMANTIC_VALUE = 0.37;

export const CASCADE_STAGES: Array<{ label: string; detail: string }> = [
  { label: "阶段一:生成语义 token", detail: "自回归生成语义 token 序列,决定内容和说话人是谁" },
  { label: "阶段二:生成粗声学 token", detail: "以语义 token 为条件,生成 RVQ 前几层粗粒度声学 token" },
  { label: "阶段三:生成细声学 token", detail: "以前两阶段为条件,生成 RVQ 剩余层的精细声学 token" },
];
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用单层模板,`AUDIOLM_SOURCE_PATH = "18-speech-audio/04-audiolm.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/SemanticTokenWidget.tsx`**

```tsx
import { useState } from "react";
import { extractSemanticToken } from "../lib/data";

export function SemanticTokenWidget() {
  const [framePos, setFramePos] = useState(0);
  const token = extractSemanticToken(framePos);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        音频帧位置(模拟不同时刻的音频内容)
        <input type="range" min={0} max={5} step={0.5} value={framePos} onChange={(e) => setFramePos(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ padding: "var(--space-4)", border: "1px solid #fb7185", borderRadius: "var(--radius-md)", background: "#fff1f2" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>语义 token id</div>
        <div style={{ fontSize: "var(--fs-2xl)", fontWeight: 700, color: "#9d174d" }}>{token}</div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        语义 token 来自自监督音频模型(w2v-BERT)的中间层表征聚类离散化,采样率较低(每个 token 覆盖更长时间跨度),携带内容和说话人身份等长程结构信息。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/SemanticTokenStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUDIOLM_SOURCE_PATH } from "../lib/prose";
import { SemanticTokenWidget } from "../widgets/SemanticTokenWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SemanticTokenStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:语义 token
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用自监督模型的中间层表征提取粗粒度语义 token,负责"接下来该说/演奏什么内容、是谁在说/演奏"。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SemanticTokenWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/RvqWidget.tsx`**

```tsx
import { useState } from "react";
import { TOY_SEMANTIC_VALUE, residualQuantize, partialReconstruction } from "../lib/data";

const W = 560;
const H = 220;

export function RvqWidget() {
  const [numLevels, setNumLevels] = useState(1);
  const { codes } = residualQuantize(TOY_SEMANTIC_VALUE);
  const recon = partialReconstruction(codes, numLevels);
  const error = Math.abs(TOY_SEMANTIC_VALUE - recon);

  const toX = (v: number) => 280 + v * 240;

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        使用的 RVQ 层数 = {numLevels}
        <input type="range" min={1} max={4} value={numLevels} onChange={(e) => setNumLevels(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`用 ${numLevels} 层 RVQ 重建,误差 ${error.toFixed(3)}`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          残差向量量化(RVQ):层数越多,重建越精确
        </text>
        <line x1={40} y1={H / 2} x2={W - 40} y2={H / 2} stroke="var(--border)" />
        <circle cx={toX(TOY_SEMANTIC_VALUE)} cy={H / 2} r={8} fill="#9ca3af" />
        <text x={toX(TOY_SEMANTIC_VALUE)} y={H / 2 - 16} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">真实值</text>
        <circle cx={toX(recon)} cy={H / 2} r={6} fill="#fb7185" />
        <text x={toX(recon)} y={H / 2 + 26} textAnchor="middle" fontSize={10} fill="#9d174d">重建值</text>
        <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
          codes = [{codes.slice(0, numLevels).join(", ")}] · 误差 = {error.toFixed(3)}
        </text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        第一层码本捕捉粗粒度信息,后续层逐层用残差方式补充更精细的细节——用的层数越多,重建值越接近真实值。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/RvqStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUDIOLM_SOURCE_PATH } from "../lib/prose";
import { RvqWidget } from "../widgets/RvqWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function RvqStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:声学 token —— 残差量化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用神经编解码器把音频压缩成多层残差量化 token,负责重建出高保真的波形细节。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={AUDIOLM_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <RvqWidget />
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
git add web/src/components/node/golden/audiolm/lib web/src/components/node/golden/audiolm/stages web/src/components/node/golden/audiolm/widgets
git commit -m "feat: AudioLM 金标本 lib + Stage1/2"
```

---

## Task 8: AudioLM — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/audiolm/widgets/CascadeWidget.tsx`
- Create: `web/src/components/node/golden/audiolm/stages/CascadeStage.tsx`
- Create: `web/src/components/node/golden/audiolm/NodePageAudioLM.tsx`
- Create: `web/src/components/node/golden/audiolm/NodePageAudioLM.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/CascadeWidget.tsx`**

```tsx
import { useState } from "react";
import { CASCADE_STAGES } from "../lib/data";

export function CascadeWidget() {
  const [completed, setCompleted] = useState(0);

  return (
    <div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {CASCADE_STAGES.map((s, i) => {
          const done = i < completed;
          const active = i === completed;
          return (
            <div
              key={s.label}
              style={{
                padding: "var(--space-3)", borderRadius: "var(--radius-md)",
                border: `1px solid ${done ? "#059669" : active ? "#fb7185" : "var(--border)"}`,
                background: done ? "#ecfdf5" : active ? "#fff1f2" : "var(--bg-surface)",
                opacity: i > completed ? 0.5 : 1,
              }}
            >
              <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: done ? "#065f46" : active ? "#9d174d" : "var(--ink-secondary)" }}>
                {done ? "✓ " : ""}{s.label}
              </div>
              <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 4 }}>{s.detail}</div>
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-4)" }}>
        <button
          type="button" onClick={() => setCompleted((c) => Math.min(c + 1, CASCADE_STAGES.length))}
          disabled={completed >= CASCADE_STAGES.length}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: completed >= CASCADE_STAGES.length ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: completed >= CASCADE_STAGES.length ? 0.5 : 1 }}
        >
          执行下一阶段
        </button>
        <button
          type="button" onClick={() => setCompleted(0)}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        三个阶段依次级联执行,每个阶段都由独立的 Transformer decoder 训练,后一阶段以前一阶段的输出为条件。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/CascadeStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUDIOLM_SOURCE_PATH } from "../lib/prose";
import { CascadeWidget } from "../widgets/CascadeWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function CascadeStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:层级式级联生成
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        语义 token → 粗声学 token → 细声学 token 三阶段级联,粗细粒度的 token 之间有清晰的条件依赖关系。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <CascadeWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageAudioLM.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageAudioLM.tsx`**

```tsx
import { Link } from "react-router";
import audiolmMarkdown from "../../../../../../18-speech-audio/04-audiolm.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, AUDIOLM_SOURCE_PATH } from "./lib/prose";
import { SemanticTokenStage } from "./stages/SemanticTokenStage";
import { RvqStage } from "./stages/RvqStage";
import { CascadeStage } from "./stages/CascadeStage";
import styles from "./NodePageAudioLM.module.css";

const prose = extractProse(audiolmMarkdown);

export default function NodePageAudioLM() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>AudioLM (2022)</h1>
        <div className={styles.metaLine}>作者:Zalán Borsos · Raphaël Marinier · Damien Vincent 等(Google)</div>
        <div className={styles.metaLine}>论文:AudioLM: a Language Modeling Approach to Audio Generation</div>
        <p className={styles.keyIdea}>
          把音频离散化成语义 token 和声学 token 两级表示,用语言模型对两级 token 做层级式 next-token 预测
        </p>
      </section>

      <section className={styles.stage}>
        <SemanticTokenStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <RvqStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <CascadeStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={AUDIOLM_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={AUDIOLM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

**注意**:hero 区域作者列表按既有惯例用"前 3 位 + 等",从 `18-speech-audio/04-audiolm.md` frontmatter 的 `authors` 数组读取确认前 3 位姓名准确(该节点共 11 位作者)。

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "18-speech-audio/04-audiolm": lazy(() => import("./audiolm/NodePageAudioLM")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/audiolm web/src/components/node/golden/index.ts
git commit -m "feat: AudioLM 金标本 Stage3 + NodePage + 注册"
```

---

## Task 9: MusicGen — lib + Stage1/2

**Files:**
- Create: `web/src/components/node/golden/musicgen/lib/data.ts`
- Create: `web/src/components/node/golden/musicgen/lib/prose.ts`
- Create: `web/src/components/node/golden/musicgen/stages/Stage.module.css`
- Create: `web/src/components/node/golden/musicgen/widgets/EncodecRvqWidget.tsx`
- Create: `web/src/components/node/golden/musicgen/stages/EncodecStage.tsx`
- Create: `web/src/components/node/golden/musicgen/widgets/DelayPatternWidget.tsx`
- Create: `web/src/components/node/golden/musicgen/stages/DelayPatternStage.tsx`

- [ ] **Step 1: 确认标题层级**

```bash
grep -n '^## ' 18-speech-audio/05-musicgen.md
```

Expected:`## 机制一:EnCodec 残差量化`、`## 机制二:码本交错(codebook interleaving)`、`## 机制三:文本 + 旋律双重条件控制`。

- [ ] **Step 2: 写 `lib/data.ts`**

```typescript
// MusicGen demo 数据:EnCodec 多层并行码本可视化 + 延迟交错模式(delay pattern)+
// 文本/旋律双重条件开关。全部确定性构造。

export const NUM_LEVELS = 4;
export const NUM_TIMESTEPS = 6;

/** 给定时间步 t 和码本层 level,生成一个确定性的码本 id(0-7),模拟 EnCodec
 * 每个时间步 NUM_LEVELS 个并行码本各自的取值。 */
export function encodecCode(t: number, level: number): number {
  let h = (t + 1) * 131 + (level + 1) * 977;
  h = h >>> 0;
  return h % 8;
}

export interface DelayCell {
  level: number;
  step: number; // 该帧位置对应的原始时间步,-1 表示 padding(尚未到达/已经结束)
  filled: boolean;
}

/** 延迟交错模式(delay pattern):第 level 层延迟 level 步开始,总长度
 * S = numTimesteps + numLevels - 1 帧,让单个自回归 Transformer 能按固定顺序
 * 逐帧预测所有层的 token,而不需要为每层单独训练模型或加阶段。 */
export function buildDelayPattern(numTimesteps: number, numLevels: number = NUM_LEVELS): DelayCell[][] {
  const S = numTimesteps + numLevels - 1;
  const rows: DelayCell[][] = [];
  for (let level = 0; level < numLevels; level++) {
    const row: DelayCell[] = [];
    for (let s = 0; s < S; s++) {
      const step = s - level;
      row.push({ level, step, filled: step >= 0 && step < numTimesteps });
    }
    rows.push(row);
  }
  return rows;
}

export type ConditionMode = "none" | "text" | "melody" | "both";

export const CONDITION_LABELS: Record<ConditionMode, string> = {
  none: "无条件(纯续写)",
  text: "仅文本条件",
  melody: "仅旋律条件",
  both: "文本 + 旋律双重条件",
};
```

- [ ] **Step 3: 写 `lib/prose.ts`**

用通用单层模板,`MUSICGEN_SOURCE_PATH = "18-speech-audio/05-musicgen.md"`。

- [ ] **Step 4: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `widgets/EncodecRvqWidget.tsx`**

```tsx
import { NUM_LEVELS, NUM_TIMESTEPS, encodecCode } from "../lib/data";

const CELL = 36;

export function EncodecRvqWidget() {
  return (
    <div>
      <svg viewBox={`0 0 ${NUM_TIMESTEPS * CELL + 80} ${NUM_LEVELS * CELL + 40}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="EnCodec 多层并行残差量化码本可视化">
        <text x={(NUM_TIMESTEPS * CELL + 80) / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          每个时间步 {NUM_LEVELS} 个并行码本(粗→细)
        </text>
        {Array.from({ length: NUM_LEVELS }, (_, level) => (
          <text key={level} x={20} y={40 + level * CELL + CELL / 2 + 4} fontSize={10} fill="var(--ink-muted)">层{level}</text>
        ))}
        {Array.from({ length: NUM_TIMESTEPS }, (_, t) =>
          Array.from({ length: NUM_LEVELS }, (_, level) => {
            const code = encodecCode(t, level);
            const lightness = 90 - level * 12;
            return (
              <g key={`${t}-${level}`}>
                <rect x={50 + t * CELL} y={30 + level * CELL} width={CELL - 2} height={CELL - 2} fill={`hsl(350, 70%, ${lightness}%)`} />
                <text x={50 + t * CELL + (CELL - 2) / 2} y={30 + level * CELL + (CELL - 2) / 2 + 4} textAnchor="middle" fontSize={10} fill="#fff">{code}</text>
              </g>
            );
          })
        )}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        每一帧时间步上,{NUM_LEVELS} 个码本并行各自贡献一个离散 token——层 0 捕捉最粗粒度的信息,层数越高补充的细节越精细。单阶段模型需要同时处理所有层,而不是像 AudioLM 那样分阶段生成。
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/EncodecStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MUSICGEN_SOURCE_PATH } from "../lib/prose";
import { EncodecRvqWidget } from "../widgets/EncodecRvqWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function EncodecStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:EnCodec 残差量化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用神经音频编解码器把音频压缩成多层残差量化码本,K 层组合起来能重建出接近原始质量的音频。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <EncodecRvqWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/DelayPatternWidget.tsx`**

```tsx
import { useState } from "react";
import { NUM_LEVELS, buildDelayPattern } from "../lib/data";

const CELL = 34;

export function DelayPatternWidget() {
  const [numTimesteps, setNumTimesteps] = useState(6);
  const rows = buildDelayPattern(numTimesteps);
  const S = numTimesteps + NUM_LEVELS - 1;

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        原始时间步数 T = {numTimesteps}(交错后总长度 = T + {NUM_LEVELS} - 1 = {S})
        <input type="range" min={4} max={8} value={numTimesteps} onChange={(e) => setNumTimesteps(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${S * CELL + 80} ${NUM_LEVELS * CELL + 40}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`延迟交错模式,T=${numTimesteps}`}>
        <text x={(S * CELL + 80) / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          延迟交错模式(delay pattern):每层延迟自己的层号开始
        </text>
        {rows.map((row, level) => (
          <g key={level}>
            <text x={20} y={40 + level * CELL + CELL / 2 + 4} fontSize={10} fill="var(--ink-muted)">层{level}</text>
            {row.map((cell, s) => (
              <rect
                key={s}
                x={50 + s * CELL}
                y={30 + level * CELL}
                width={CELL - 2}
                height={CELL - 2}
                fill={cell.filled ? "#fb7185" : "var(--bg-subtle)"}
                stroke="var(--bg-canvas)"
              />
            ))}
          </g>
        ))}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        粉色格子是真实 token,灰色是 padding。每层沿对角线错开一步,让单个自回归 Transformer 按固定顺序逐帧预测,同时覆盖所有层——不需要为每层单独训练模型或加阶段。
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/DelayPatternStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MUSICGEN_SOURCE_PATH } from "../lib/prose";
import { DelayPatternWidget } from "../widgets/DelayPatternWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DelayPatternStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:码本交错(codebook interleaving)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把多个并行码本流按延迟错位规则重新排列成一条单一序列,单阶段模型就能同时建模所有层。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={MUSICGEN_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <DelayPatternWidget />
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
git add web/src/components/node/golden/musicgen/lib web/src/components/node/golden/musicgen/stages web/src/components/node/golden/musicgen/widgets
git commit -m "feat: MusicGen 金标本 lib + Stage1/2"
```

---

## Task 10: MusicGen — Stage3 + NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/musicgen/widgets/ConditionToggleWidget.tsx`
- Create: `web/src/components/node/golden/musicgen/stages/ConditionStage.tsx`
- Create: `web/src/components/node/golden/musicgen/NodePageMusicGen.tsx`
- Create: `web/src/components/node/golden/musicgen/NodePageMusicGen.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/ConditionToggleWidget.tsx`**

```tsx
import { useState } from "react";
import { ConditionMode, CONDITION_LABELS } from "../lib/data";

export function ConditionToggleWidget() {
  const [textOn, setTextOn] = useState(true);
  const [melodyOn, setMelodyOn] = useState(false);

  const mode: ConditionMode = textOn && melodyOn ? "both" : textOn ? "text" : melodyOn ? "melody" : "none";

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: "var(--space-4)" }}>
        <button
          type="button" onClick={() => setTextOn((v) => !v)} aria-pressed={textOn}
          style={{
            padding: "6px 16px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${textOn ? "#fb7185" : "var(--border)"}`,
            background: textOn ? "#fb7185" : "var(--bg-surface)",
            color: textOn ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          文本条件(T5 编码)
        </button>
        <button
          type="button" onClick={() => setMelodyOn((v) => !v)} aria-pressed={melodyOn}
          style={{
            padding: "6px 16px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${melodyOn ? "#fb7185" : "var(--border)"}`,
            background: melodyOn ? "#fb7185" : "var(--bg-surface)",
            color: melodyOn ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          旋律条件(chromagram)
        </button>
      </div>
      <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>当前生成模式</div>
        <div style={{ fontSize: "var(--fs-lg)", fontWeight: 700, color: "#9d174d" }}>{CONDITION_LABELS[mode]}</div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        文本条件通过 T5 文本编码器以 cross-attention 方式注入;旋律条件从参考音频提取色度图(音高/和声走向)。两种条件可以单独或组合使用。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/ConditionStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MUSICGEN_SOURCE_PATH } from "../lib/prose";
import { ConditionToggleWidget } from "../widgets/ConditionToggleWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ConditionStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:文本 + 旋律双重条件控制
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        文本条件控制生成音乐的风格/描述内容,旋律条件让模型按指定旋律生成不同编曲风格的音乐,两种条件可单独或组合使用。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <ConditionToggleWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageMusicGen.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageMusicGen.tsx`**

```tsx
import { Link } from "react-router";
import musicgenMarkdown from "../../../../../../18-speech-audio/05-musicgen.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, MUSICGEN_SOURCE_PATH } from "./lib/prose";
import { EncodecStage } from "./stages/EncodecStage";
import { DelayPatternStage } from "./stages/DelayPatternStage";
import { ConditionStage } from "./stages/ConditionStage";
import styles from "./NodePageMusicGen.module.css";

const prose = extractProse(musicgenMarkdown);

export default function NodePageMusicGen() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>MusicGen (2023)</h1>
        <div className={styles.metaLine}>作者:Jade Copet · Felix Kreuk · Itai Gat · Tal Remez · David Kant · Gabriel Synnaeve · Yossi Adi · Alexandre Défossez</div>
        <div className={styles.metaLine}>论文:Simple and Controllable Music Generation</div>
        <p className={styles.keyIdea}>
          单阶段 Transformer decoder + EnCodec 码本交错技巧,把 AudioLM/MusicLM 的多阶段级联简化成单阶段模型
        </p>
      </section>

      <section className={styles.stage}>
        <EncodecStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DelayPatternStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ConditionStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={MUSICGEN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={MUSICGEN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "18-speech-audio/05-musicgen": lazy(() => import("./musicgen/NodePageMusicGen")),
```

- [ ] **Step 6: 跑测试**

```bash
cd web
npx vitest run AllGoldenSamples.smoke ProseCompleteness
npx tsc --noEmit
```

- [ ] **Step 7: 提交**

```bash
git add web/src/components/node/golden/musicgen web/src/components/node/golden/index.ts
git commit -m "feat: MusicGen 金标本 Stage3 + NodePage + 注册"
```

---

## Task 11: 全项目验证

**Files:** 无新建/修改,纯验证任务。

- [ ] **Step 1: 跑全量 vitest**

```bash
cd web && npx vitest run
```

Expected: 全部通过,`AllGoldenSamples.smoke.test.tsx` 新增 5 条冒烟用例 PASS,`ProseCompleteness.test.tsx` 新增 5 个 prose 模块用例 PASS,总用例数比 Task 开始前(339)多至少 10 条。

- [ ] **Step 2: 跑 tsc**

```bash
cd web && npx tsc --noEmit
```

Expected: 无错误。

- [ ] **Step 3: 确认全部 84 个节点都有金标本(补完这批后应无遗漏)**

```bash
python3 -c "
import json, re
data = json.load(open('web/src/data/families.json'))
with open('web/src/components/node/golden/index.ts') as f:
    content = f.read()
registered = set(re.findall(r'\"([^\"]+)\":\s*lazy', content))
total = 0
missing = []
for fam in data['families']:
    for n in fam['nodes']:
        key = f\"{fam['id']}/{n['path'].split('/')[-1].replace('.md','')}\"
        total += 1
        if key not in registered:
            missing.append(key)
print(f'registered={len(registered)} total={total}')
if missing:
    print('仍缺:', missing)
else:
    print('全部节点都已有金标本交互页')
"
```

Expected: `registered=84 total=84`,`全部节点都已有金标本交互页`。

- [ ] **Step 4: 浏览器验证**

用 preview_start 起 web/ 的 dev server,依次打开并确认交互式 Stage 渲染(非纯 markdown 兜底)、核心交互有响应、console 无 error:

- `/families/18-speech-audio/01-wav2vec2`
- `/families/18-speech-audio/02-hubert`
- `/families/18-speech-audio/03-whisper`
- `/families/18-speech-audio/04-audiolm`
- `/families/18-speech-audio/05-musicgen`

- [ ] **Step 5: 提交(若浏览器验证发现小问题并修复)**

```bash
git add -A
git commit -m "fix: 语音家族金标本浏览器验证发现的问题修复"
```

(若验证全部通过、无需修复,跳过此步)

---

## Plan Self-Review 记录

- **Spec 覆盖**:spec 第 3 节列出的 5 个节点、每节点 3 个 Stage 的设计要点均对应到 Task 1-10 里的具体 Stage/Widget;spec 第 6 节验收标准对应 Task 11。
- **prose 模板一致性**:确认本家族用 GNN 式扁平单层模板(与 World Models 轮的双层模板不同),已在"关键背景"里给出完整代码,5 个节点直接复用,只改 `XXX_SOURCE_PATH` 常量。
- **数学正确性预验证**:`residualQuantize`/`partialReconstruction`(AudioLM/MusicGen 共用的 RVQ 设计思路)、`buildDelayPattern`(MusicGen,复用此前 markdown 撰写阶段已验证过的 S=T+K-1 交错逻辑)、`updateCenters`(HuBERT,标准 Lloyd's 算法保证收敛)、`classDistribution`(HuBERT,true class 恒定 logit 更高,任意 sharpness 下都是最高概率)在设计阶段已给出正确性论证,实现时精确复制公式。
- **Placeholder 扫描**:所有步骤均为完整可运行代码,无 TBD/TODO。
- **类型一致性**:`ProseSections` 接口在"关键背景"通用模板中统一定义一次,5 个节点的 `lib/prose.ts` 都从这个模板复制,字段名在所有 Stage 组件的 props 里保持一致引用。
