import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LEARNING_TO_SUMMARIZE_SOURCE_PATH } from "../lib/prose";
import { PIPELINE_STAGES } from "../lib/data";
import { PreferenceComparisonDiagram } from "../widgets/PreferenceComparisonDiagram";
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

export function PreferenceCollectionStage({ intuitionProse, mechanism1Prose }: Props) {
  const [winner, setWinner] = useState<"A" | "B">("A");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:三阶段 RLHF 流程 — 收集人类偏好比较
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        2020 年前 NLP 生成任务靠 (输入, 参考摘要) 监督微调 + ROUGE 评估,但 ROUGE
        和人类判断相关性差、监督学习只能模仿参考。Stiennon 等人反过来:让 SFT
        模型对同一篇 Reddit 帖子生成多个候选摘要,标注员两两比较选出更好的一个 ——
        这份偏好数据是整个 RLHF 流程的起点。
      </p>

      <PreferenceComparisonDiagram winner={winner} />
      <p className={styles.caption}>
        ↑ 切换看标注员选择候选 A 还是候选 B 作为 winner。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setWinner("A")} style={btnStyle(winner === "A")}>候选 A 更好</button>
        <button type="button" onClick={() => setWinner("B")} style={btnStyle(winner === "B")}>候选 B 更好</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              三阶段流程速览
            </div>
            {PIPELINE_STAGES.map((s, i) => (
              <div key={s.key} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: i < PIPELINE_STAGES.length - 1 ? "1px dashed var(--border)" : "none" }}>
                <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "var(--ink-primary)" }}>{i + 1}. {s.label}</div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 2, lineHeight: 1.5 }}>{s.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
