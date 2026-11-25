import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TOOLFORMER_SOURCE_PATH } from "../lib/prose";
import { CandidateSamplingDiagram } from "../widgets/CandidateSamplingDiagram";
import { ToolsGridDiagram } from "../widgets/ToolsGridDiagram";
import { CANDIDATE_EXAMPLES } from "../lib/data";
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

export function SamplingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [idx, setIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:LM 自采样候选 — 用 few-shot 让 LLM 当标注员
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ReAct 的 tool use 完全靠 prompt 教,在小模型上不 work。Toolformer
        反过来:不用 prompt 教,让 LLM 自监督学。给定文本,LM 看 ≤20 个
        手写例子后,自己判断"该不该在这里插调用、插什么调用" — 整个语料
        百万级,人工标不动,但 LLM 自己 prompt 自己,几乎零成本生成海量候选。
      </p>

      <CandidateSamplingDiagram idx={idx} />
      <p className={styles.caption}>
        ↑ 切换看不同句子的候选采样结果 — 只有 P(插入调用) 超过阈值 τ 才会被采样。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        {CANDIDATE_EXAMPLES.map((ex, i) => (
          <button key={ex.sentence} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>
            例 {i + 1}
          </button>
        ))}
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <ToolsGridDiagram />
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={TOOLFORMER_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={TOOLFORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              与 ReAct 的根本区别
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              ReAct 靠 prompt 教,在 GPT-3.5+ 上 work,但小模型能力不够。
              Toolformer 把 tool use 从 prompt-level 技巧降到
              pretraining-level 能力 — 微调后模型"学到调工具"就像"学到
              用某个词"一样自然,不需要每次都靠 prompt 提醒。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
