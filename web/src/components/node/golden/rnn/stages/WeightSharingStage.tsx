import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RNN_SOURCE_PATH } from "../lib/prose";
import { WeightSharingChart } from "../widgets/WeightSharingChart";
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

export function WeightSharingStage({ mechanism2Prose }: Props) {
  const [seqLen, setSeqLen] = useState(50);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:权重共享 — 同一组 (W_x, W_h, W_y) 复用所有 T 步
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        RNN 所有时间步用同一组权重,这是它能处理变长序列的根本 — 和 CNN 卷积核在空间上扫
        完全同构,只是共享方向从"空间"换成"时间"。参数量与序列长度无关,
        训练时见过 50 步,推理时可以输入 200 步(虽然质量未必跟得上)。
      </p>

      <WeightSharingChart seqLen={seqLen} />
      <p className={styles.caption}>
        ↑ 绿色共享权重曲线是平的,粉色假想不共享曲线随长度线性增长。
        拖动看不同序列长度下参数量差距。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>序列长度 T</span><strong>{seqLen}</strong>
      </label>
      <input type="range" min={5} max={200} step={5} value={seqLen}
             onChange={(e) => setSeqLen(parseInt(e.target.value))} style={{ width: "100%" }} />
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        {[10, 50, 100, 200].map((l) => (
          <button key={l} type="button" onClick={() => setSeqLen(l)} style={btnStyle(seqLen === l)}>T={l}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={RNN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              思想传承
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              "权重共享让模型尺寸独立于序列长度"这一设计后来被 Transformer 完全继承 —
              不管输入 token 是 100 还是 100K,Transformer 的参数都不变
              (虽然 attention 的 KV cache 与长度成正比,那是 inference 时的工程问题)。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
