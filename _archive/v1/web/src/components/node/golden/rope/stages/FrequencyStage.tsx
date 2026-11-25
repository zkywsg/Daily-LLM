import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ROPE_SOURCE_PATH } from "../lib/prose";
import { FreqDecayChart } from "../widgets/FreqDecayChart";
import { MultiPlaneDiagram } from "../widgets/MultiPlaneDiagram";
import { buildFreqDims } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function FrequencyStage({ mechanism2Prose }: Props) {
  const [freqIdx, setFreqIdx] = useState(-1);
  const dims = buildFreqDims();

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:多频率拆分 — d/2 个 2D 平面 + 长程衰减
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        实际 d_head 是几十到上百维,RoPE 拆成 d/2 个 2D 平面,每个用不同频率
        θ_i = 10000^(-2(i-1)/d)。高频维度旋转快,只能捕捉短距依赖;低频维度旋转慢,
        捕捉长距依赖。所有维度内积求和后,远距离时在不同维度上"散开"成不相关的旋转,
        自然产生长程衰减,不需要显式 attention mask。
      </p>

      <MultiPlaneDiagram highlightIdx={freqIdx} />
      <p className={styles.caption}>
        ↑ 同一位置 m=5 在不同频率平面上旋转角度不同。点按钮聚焦某个频率维度。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setFreqIdx(-1)} style={btnStyle(freqIdx === -1)}>全部</button>
        {dims.map((f, i) => (
          <button key={i} type="button" onClick={() => setFreqIdx(i)} style={btnStyle(freqIdx === i)}>{f.name}</button>
        ))}
      </div>

      <FreqDecayChart highlightFreq={freqIdx >= 0 ? dims[freqIdx].index : null} />
      <p className={styles.caption}>
        ↑ 多个频率求和后的内积随相对距离振荡衰减 — "远的词关系弱"自然涌现。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={ROPE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              O(d) 高效实现
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]  # 偶数项
    x2 = x[..., 1::2]  # 奇数项
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return rotated.flatten(-2)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              不需要构造 d×d 旋转矩阵,利用稀疏结构直接向量化。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
