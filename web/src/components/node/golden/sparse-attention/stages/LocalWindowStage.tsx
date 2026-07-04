import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SPARSE_ATTN_SOURCE_PATH } from "../lib/prose";
import { AttentionMatrixDiagram } from "../widgets/AttentionMatrixDiagram";
import { LocalWindowDiagram } from "../widgets/LocalWindowDiagram";
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

export function LocalWindowStage({ intuitionProse, mechanism1Prose }: Props) {
  const [showLocal, setShowLocal] = useState(true);
  const [showGlobal, setShowGlobal] = useState(true);
  const [showRandom, setShowRandom] = useState(true);
  const [centerIdx, setCenterIdx] = useState(10);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Local 滑窗 — 每个 token 看周围 w 个邻居
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        原版 Transformer attention 是 O(N²),16K 上下文要算 2.7 亿个分数,QK^T 矩阵
        物化要 32GB 显存,直接超出 A100。但 dense attention 算完后你会发现大多数
        (i,j) 位置权重接近 0 — token 关心的几乎都是局部邻居。Local 滑窗把每个位置
        只 attend 到 [i-w/2, i+w/2] 这 w 个邻居,复杂度 O(N×w)=O(N)。
      </p>

      <AttentionMatrixDiagram showLocal={showLocal} showGlobal={showGlobal} showRandom={showRandom} />
      <p className={styles.caption}>
        ↑ N×N attention 矩阵可视化,勾选/取消看三类稀疏连接如何组合成完整模式。
      </p>
      <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "var(--fs-sm)" }}>
          <input type="checkbox" checked={showLocal} onChange={(e) => setShowLocal(e.target.checked)} /> Local
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "var(--fs-sm)" }}>
          <input type="checkbox" checked={showGlobal} onChange={(e) => setShowGlobal(e.target.checked)} /> Global
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "var(--fs-sm)" }}>
          <input type="checkbox" checked={showRandom} onChange={(e) => setShowRandom(e.target.checked)} /> Random
        </label>
      </div>

      <LocalWindowDiagram windowSize={6} centerIdx={centerIdx} />
      <p className={styles.caption}>
        ↑ 1D 序列上的滑窗演示。拖动看查询位置切换时窗口跟随移动。
      </p>
      <input type="range" min={0} max={19} step={1} value={centerIdx}
             onChange={(e) => setCenterIdx(parseInt(e.target.value))} style={{ width: "100%", marginTop: 8 }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              Dilated Sliding Window
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              像 CNN dilated convolution 一样,每隔 d 个位置取一个邻居 —
              同样 w 个 attention 邻居但有效感受野扩大 d 倍。
              这一思想在 Swin Transformer 的 "shifted window" 里被进一步推广。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
