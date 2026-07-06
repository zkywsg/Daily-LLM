import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { REACT_SOURCE_PATH } from "../lib/prose";
import { ToolCallDiagram } from "../widgets/ToolCallDiagram";
import { TOOLS } from "../lib/data";
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

export function ActionStage({ mechanism2Prose }: Props) {
  const [idx, setIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Action — 让 LLM 接入真实世界
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Action 是 ReAct 闭环的"手"。LLM 不再仅靠参数化知识,而是调真正的
        外部工具。工具集很简单:Search 查 Wikipedia、Lookup 在当前页面找、
        Calculator 算数、Finish 终止。关键不是工具复杂,而是 LLM 可以根据
        Thought 决定调哪个工具,以及参数怎么填。
      </p>

      <ToolCallDiagram selectedIdx={idx} />
      <p className={styles.caption}>
        ↑ 点按钮切换看 Thought 如何指向不同工具。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {TOOLS.map((t, i) => (
          <button key={t.name} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{t.name}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={REACT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              工具集
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              {TOOLS.map((t) => (
                <li key={t.name}><code>{t.name}</code> — {t.desc}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
