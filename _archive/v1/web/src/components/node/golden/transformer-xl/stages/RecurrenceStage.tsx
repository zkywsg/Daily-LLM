import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TRANSFORMER_XL_SOURCE_PATH } from "../lib/prose";
import { SegmentRecurrenceDiagram } from "../widgets/SegmentRecurrenceDiagram";
import { EffectiveContextChart } from "../widgets/EffectiveContextChart";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function RecurrenceStage({ intuitionProse, mechanism1Prose }: Props) {
  const [currentSegment, setCurrentSegment] = useState(2);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Segment-Level Recurrence — 段内并行 + 段间循环
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        原版 Transformer 处理长文本时把它切成不重叠的固定窗口段,每段独立训练——
        段边界处"零记忆",跨段依赖完全丢失。Transformer-XL 把上一段每层隐状态
        stop-gradient 缓存下来,作为当前段 attention 的 K/V 前缀拼进去:段内仍然
        完全并行,段之间通过缓存"接力",像 RNN 一样跨段传递信息但不牺牲并行性。
      </p>

      <SegmentRecurrenceDiagram currentSegment={currentSegment} totalSegments={4} />
      <p className={styles.caption}>
        ↑ 拖动滑块切换"当前处理到第几段",看当前段(蓝色实线)如何 attend 到已缓存的历史段(灰色虚线,stop-gradient)。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>当前段</span><strong>第 {currentSegment} 段</strong>
      </label>
      <input
        type="range"
        min={1}
        max={4}
        step={1}
        value={currentSegment}
        onChange={(e) => setCurrentSegment(parseInt(e.target.value))}
        style={{ width: "100%" }}
      />
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        {[1, 2, 3, 4].map((s) => (
          <button key={s} type="button" onClick={() => setCurrentSegment(s)} style={btnStyle(currentSegment === s)}>
            第 {s} 段
          </button>
        ))}
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <EffectiveContextChart />
        <p className={styles.caption}>
          ↑ 论文实测有效上下文从 512 → 3800 token(7.4×),因为信息逐层向上累积,理论最大上下文 O(段数 × 层数)。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              SG(·) = stop-gradient
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`h~_{τ+1} = [SG(h_τ); h_{τ+1}]
Q_{τ+1} = h_{τ+1} W_Q
K_{τ+1} = h~_{τ+1} W_K
V_{τ+1} = h~_{τ+1} W_V`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              Q 只来自当前段,K/V 来自"缓存 + 当前"— 信息单向流动,梯度不跨段回传,训练成本不会爆炸。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
