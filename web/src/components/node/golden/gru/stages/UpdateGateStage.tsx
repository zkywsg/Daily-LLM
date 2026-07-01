import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRU_SOURCE_PATH } from "../lib/prose";
import { ConvexCombination } from "../widgets/ConvexCombination";
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

export function UpdateGateStage({ mechanism2Prose }: Props) {
  const [z, setZ] = useState(0.4);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:更新门 z — 凸组合替代 LSTM 独立 forget + input
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        更新门 z_t 同时承担 LSTM 的 forget gate 和 input gate 职责:
        h_t = (1-z_t)⊙h_{"{t-1}"} + z_t⊙h̃_t 是一个凸组合,(1-z)+z=1 自动满足。
        LSTM 的 f 和 i 独立学习,理论上能做到"既忘又不写"(清空状态),
        GRU 用凸组合绑定后失去这一能力,但实践上几乎不损害性能。
      </p>

      <ConvexCombination z={z} />
      <p className={styles.caption}>
        ↑ 拖动 z 看旧状态(蓝)和新候选(粉)如何按比例混合。
        下方对比 LSTM 的独立门 — f 和 i 可以取任意组合,GRU 做不到这点。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>更新门 z</span><strong>{z.toFixed(2)}</strong>
      </label>
      <input type="range" min={0} max={1} step={0.05} value={z}
             onChange={(e) => setZ(parseFloat(e.target.value))} style={{ width: "100%" }} />
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button type="button" onClick={() => setZ(0)} style={btnStyle(z === 0)}>z=0 全保留</button>
        <button type="button" onClick={() => setZ(0.5)} style={btnStyle(z === 0.5)}>z=0.5 平均</button>
        <button type="button" onClick={() => setZ(1)} style={btnStyle(z === 1)}>z=1 全替换</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GRU_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              凸组合 vs 独立门
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>状态组合</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>LSTM (f,i)</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>GRU ((1-z),z)</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>全忘+全写</td>
                  <td style={{ padding: "8px" }}>✓ (0,1)</td>
                  <td style={{ padding: "8px" }}>✓ z=1</td>
                </tr>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>全保留+不写</td>
                  <td style={{ padding: "8px" }}>✓ (1,0)</td>
                  <td style={{ padding: "8px" }}>✓ z=0</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px" }}>既忘又不写(清空)</td>
                  <td style={{ padding: "8px" }}>✓ (0,0)</td>
                  <td style={{ padding: "8px", color: "#ec4899" }}>✗ 做不到</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
