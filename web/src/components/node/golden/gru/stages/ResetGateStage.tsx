import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRU_SOURCE_PATH } from "../lib/prose";
import { ResetGateDiagram } from "../widgets/ResetGateDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ResetGateStage({ intuitionProse, mechanism1Prose }: Props) {
  const [r, setR] = useState(0.7);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:重置门 r — 决定候选状态用多少历史
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        LSTM 三门里 forget 和 input 看起来互补(忘多少+写多少≈1),cell state C 和 hidden h
        也有信息重叠。Cho 等人 2014 反问:能不能把这些冗余合并掉?重置门 r 是第一步 —
        决定计算候选 h̃ 时用多少上一时刻的 h_{"{t-1}"}。r≈0 完全忽略历史(新序列起点),
        r≈1 完全使用历史。
      </p>

      <ResetGateDiagram r={r} />
      <p className={styles.caption}>
        ↑ 拖动 r 看历史信息(蓝色框透明度)如何被调制后送入候选 h̃ 的计算。
        注意 r 只影响新写入内容,不直接作用在 h_t 本身上。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>重置门 r</span><strong>{r.toFixed(2)}</strong>
      </label>
      <input type="range" min={0} max={1} step={0.05} value={r}
             onChange={(e) => setR(parseFloat(e.target.value))} style={{ width: "100%" }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GRU_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GRU_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              公式
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`r_t = σ(W_r [h_{t-1}, x_t] + b_r)

h̃_t = tanh(W_h [r_t⊙h_{t-1}, x_t] + b_h)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              给模型"在某个 step 重启上下文"的能力 — 比如句子边界、话题切换处。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
