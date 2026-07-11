import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SELF_CONSISTENCY_SOURCE_PATH } from "../lib/prose";
import { AnswerMarginalizationDiagram } from "../widgets/AnswerMarginalizationDiagram";
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

export function AnswerNormalizationStage({ mechanism2Prose }: Props) {
  const [normalized, setNormalized] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Answer Normalization — 让不同表达对齐
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        不同推理路径可能用不同表达方式给出同一个答案 —— "196"、"196 升"、
        "the answer is 196"、"196 liters"。如果不归一化,投票会把这些当成
        4 个不同答案,谁都占不到多数。答案提取必须正则化到统一格式,才能让
        投票机制真正生效。
      </p>

      <AnswerMarginalizationDiagram normalized={normalized} />
      <p className={styles.caption}>
        ↑ 切换看归一化前后,同一个数字答案的不同表达方式如何被投票机制处理。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setNormalized(true)} style={btnStyle(normalized)}>已归一化</button>
        <button type="button" onClick={() => setNormalized(false)} style={btnStyle(!normalized)}>未归一化</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              归一化正则(简化版)
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`def normalize(answer):
    nums = re.findall(
        r"-?\\d+(?:\\.\\d+)?", answer)
    return nums[-1] if nums else None`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              只保留最后一个数字,丢弃单位与语言差异 —— "196 升"和"the answer is 196"都归到 "196"。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
