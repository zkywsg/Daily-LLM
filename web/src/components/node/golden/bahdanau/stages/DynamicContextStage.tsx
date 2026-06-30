import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BAHDANAU_SOURCE_PATH } from "../lib/prose";
import { DynamicContextBars } from "../widgets/DynamicContextBars";
import { ALIGNMENT_DEMO } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
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

export function DynamicContextStage({ mechanism3Prose, synergyProse }: Props) {
  const [tStep, setTStep] = useState(4); // 默认显示 "zone"

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Context Vector — 按权重加和后输入 decoder
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        有了 α_t,把 encoder 隐状态按权重加和得 c_t = Σ α_{`{t,i}`} · h_i。
        decoder 用 c_t 替代原来固定 c · s_t = GRU(s_{`{t-1}`}, [y_{`{t-1}`}; c_t])。
        关键是 c_t 随 t 变化 — 每一步按需从 encoder 拉最相关的信息。
      </p>

      <DynamicContextBars tStep={tStep} />
      <p className={styles.caption}>
        ↑ 切换目标词位置看 α 分布如何动态聚焦不同源词。
        译 "zone" 时聚焦 "Area",译 "économique" 时聚焦 "Economic" — 反序对齐自动学到。
      </p>

      <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          目标词序号 t — 当前译第 {tStep + 1} 个 "{ALIGNMENT_DEMO.tgt[tStep]}"
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {ALIGNMENT_DEMO.tgt.slice(0, 13).map((t, i) => (
            <button key={i} type="button" onClick={() => setTStep(i)} style={btnStyle(i === tStep)}>{i + 1}·{t}</button>
          ))}
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={BAHDANAU_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={BAHDANAU_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              三件套 → Transformer 对应
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>Bahdanau 2014</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>Transformer 2017</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>Encoder 全输出 (BiRNN)</td>
                  <td style={{ padding: "8px" }}>Self-attention 每层全互看</td>
                </tr>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>Additive score (MLP)</td>
                  <td style={{ padding: "8px" }}><code>softmax(QKᵀ/√d_k)</code></td>
                </tr>
                <tr>
                  <td style={{ padding: "8px" }}>动态 c_t = Σ α h_i</td>
                  <td style={{ padding: "8px" }}>所有 query 一次矩阵乘并行</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, fontStyle: "italic", lineHeight: 1.5 }}>
              Vaswani 2017 标题里 "Attention" 来自这篇 2014 论文。
              Transformer 不是从天上掉下来的 — 它是把 Bahdanau 推到逻辑终点的产物。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
