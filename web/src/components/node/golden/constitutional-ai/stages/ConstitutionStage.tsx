import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CONSTITUTIONAL_AI_SOURCE_PATH } from "../lib/prose";
import { CONSTITUTION_PRINCIPLES } from "../lib/data";
import { SelfCritiqueRevisionDiagram } from "../widgets/SelfCritiqueRevisionDiagram";
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

export function ConstitutionStage({ intuitionProse, mechanism1Prose }: Props) {
  const [principleId, setPrincipleId] = useState(CONSTITUTION_PRINCIPLES[0].id);
  const principle = CONSTITUTION_PRINCIPLES.find((p) => p.id === principleId) ?? CONSTITUTION_PRINCIPLES[0];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Constitution — 16 条书面原则,自然语言定义"想要的行为"
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        RLHF 对齐依赖人工标注员按"心中模糊标准"判断什么是有害的,标准会漂移、会
        因人而异。Constitutional AI 用一份书面 constitution(约 16 条原则)替代
        这套模糊标准 —— 原则故意写得模糊高层(不列举具体违规类型),迫使 LLM 学到
        判断准则而不是查表匹配。
      </p>

      <SelfCritiqueRevisionDiagram principle={principle} />
      <p className={styles.caption}>
        ↑ 切换不同 constitution 原则,看 AI 的 critique 措辞如何随原则改变 —— 同一个有害回答,
        按不同原则会得到不同角度的批评。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {CONSTITUTION_PRINCIPLES.map((p) => (
          <button key={p.id} type="button" onClick={() => setPrincipleId(p.id)} style={btnStyle(p.id === principleId)}>
            {p.label}
          </button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              当前原则(英文原文摘录)
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-primary)", margin: 0, lineHeight: 1.6, fontStyle: "italic" }}>
              "{principle.principle}"
            </p>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5 }}>
              原则故意写得模糊高层 —— 没有列举具体的有害类型,迫使 LLM 学到判断准则本身,
              而不是查表匹配已知的违规模式。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
