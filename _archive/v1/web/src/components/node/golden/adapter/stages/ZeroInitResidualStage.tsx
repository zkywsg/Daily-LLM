import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ADAPTER_SOURCE_PATH } from "../lib/prose";
import { ZERO_INIT_FRAMES } from "../lib/data";
import { ZeroInitBehaviorDiagram } from "../widgets/ZeroInitBehaviorDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function ZeroInitResidualStage({ mechanism2Prose }: Props) {
  const [phase, setPhase] = useState<"before" | "after">("before");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Residual + 零初始化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Bottleneck 结构决定了参数量,但插入一个随机初始化的模块会立刻打乱预训练
        表示。W_up 初始化为 0,配合 residual,让 adapter 在训练一开始就是恒等
        映射(h ≈ x),再随训练平滑学出任务特化的偏移量 Δ。
      </p>

      <ZeroInitBehaviorDiagram phase={phase} />
      <p className={styles.caption}>
        ↑ 切换看训练开始(W_up ≈ 0,h ≈ x)与训练收敛(W_up 学到权重,h = x + Δ)
        的差异。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setPhase("before")} style={btnStyle(phase === "before")}>
          训练开始(零初始化)
        </button>
        <button type="button" onClick={() => setPhase("after")} style={btnStyle(phase === "after")}>
          训练收敛(学到偏移)
        </button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={ADAPTER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              当前状态:{ZERO_INIT_FRAMES[phase].label}
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              {ZERO_INIT_FRAMES[phase].description}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
