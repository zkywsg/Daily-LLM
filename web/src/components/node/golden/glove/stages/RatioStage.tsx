import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GLOVE_SOURCE_PATH } from "../lib/prose";
import { ProbeRatioChart } from "../widgets/ProbeRatioChart";
import { PROBE_TABLE } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
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

export function RatioStage({ intuitionProse, mechanism1Prose }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Log 共现拟合 — v_i·v_j + b_i + b_j = log X_ij
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Word2Vec 用局部窗口预测任务学词向量,理论上有点绕;count-based 路线直接分解
        共现矩阵理论清晰但效果不如 Word2Vec。Pennington 等人反问:count-based 没胜,
        不是矩阵分解本身的问题,而是分解错了对象 — 应该分解共现概率的比值,
        而非共现次数本身。比值天然消除绝对频次影响,只保留"区分能力"。
      </p>

      <ProbeRatioChart highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ ice/steam 与 4 个探针词的共现概率比值。solid 强偏 ice(8.9),
        gas 强偏 steam(0.085),water/fashion 都中性(≈1)。点按钮聚焦某一行。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {PROBE_TABLE.map((p, i) => (
          <button key={i} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{p.probe}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GLOVE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GLOVE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              推导四步
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`1. 想要 F(v_i,v_j,v_k) = P_ik/P_jk
2. 假设 F 仅依赖向量差和探针
3. 取 F = exp,得 v_i^T v_k = log P_ik
4. -log X_i 吸收为 bias:

   v_i^T v_j + b_i + b_j = log X_ij`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              这一目标的清晰性是 GloVe 相对 Word2Vec NEG 的理论优势 —
              NEG 是工程 trick,GloVe 是从假设出发的封闭推导。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
