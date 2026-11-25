import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RNN_SOURCE_PATH } from "../lib/prose";
import { UnfoldDiagram } from "../widgets/UnfoldDiagram";
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

export function HiddenStateStage({ intuitionProse, mechanism1Prose }: Props) {
  const [folded, setFolded] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:隐状态递推 — h_t = f(W_x x_t + W_h h_{"{t-1}"})
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        MLP 只能处理固定大小输入,语言/语音/时间序列这些变长数据完全进不去。
        Jordan(1986)和 Elman(1990)反问:为什么不让网络维护一个内部状态,
        每读一个新输入就更新它,把所有历史信息压在状态里带着走?
        Elman 反馈上一时刻的隐状态(而非 Jordan 的输出),后续所有循环网络都沿用这一路线。
      </p>

      <UnfoldDiagram folded={folded} />
      <p className={styles.caption}>
        ↑ 循环表示 vs 按时间展开。展开后 4 个时间步共享同一组权重,
        等价于深度 T 的前馈网络,可以跑标准反传。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setFolded(true)} style={btnStyle(folded)}>循环表示</button>
        <button type="button" onClick={() => setFolded(false)} style={btnStyle(!folded)}>按时间展开</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={RNN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={RNN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              Jordan vs Elman
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>方案</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>反馈源</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>Jordan</td>
                  <td style={{ padding: "8px" }}>输出 y_{"{t-1}"}</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px", fontWeight: 700, color: "#ec4899" }}>Elman(胜出)</td>
                  <td style={{ padding: "8px", fontWeight: 700, color: "#ec4899" }}>隐状态 h_{"{t-1}"}</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              隐状态维度更高、信息更丰富 — 这是后续所有循环网络沿用 Elman 路线的原因。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
