import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FASTTEXT_SOURCE_PATH } from "../lib/prose";
import { SubwordSumDiagram } from "../widgets/SubwordSumDiagram";
import { OOV_EXAMPLES } from "../lib/data";
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

export function VectorSumStage({ mechanism2Prose }: Props) {
  const [idx, setIdx] = useState(0);
  const example = OOV_EXAMPLES[idx];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:词向量 = subword 向量之和
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        每个 n-gram g 有自己的向量 z_g,词 w 的向量 v_w = Σ z_g。训练目标延续
        Word2Vec skip-gram + negative sampling,但相似度函数把"词向量内积"换成
        "subword 向量和的内积"。这一改动的直接副产品:测试时遇到没见过的词,
        只用它的 subword 组合出向量 — OOV 问题被自然解决。
      </p>

      <SubwordSumDiagram word={example.word} showOov={idx > 0} />
      <p className={styles.caption}>
        ↑ {example.note}
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {OOV_EXAMPLES.map((e, i) => (
          <button key={e.word} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>
            {e.word}({e.label})
          </button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={FASTTEXT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              subword 求和解决的三类问题
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>形态共享</strong>:"ev"、"evler"、"evlerim" 共享核心 subword,向量自然相似</li>
              <li><strong>罕见词</strong>:罕见词的 subword 在常见词中频繁出现,训练充分</li>
              <li><strong>typo 鲁棒</strong>:"apple" vs "applle" 共享 9/11 个 subword,向量近似</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
