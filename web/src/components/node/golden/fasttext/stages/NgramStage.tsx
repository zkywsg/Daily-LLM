import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FASTTEXT_SOURCE_PATH } from "../lib/prose";
import { NgramDecomposeDiagram } from "../widgets/NgramDecomposeDiagram";
import { DEMO_WORDS } from "../lib/data";
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

export function NgramStage({ intuitionProse, mechanism1Prose }: Props) {
  const [word, setWord] = useState(DEMO_WORDS[0]);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Subword n-gram 分解 — 短 + 长 + 整词三层覆盖
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Word2Vec / GloVe 把词当作原子单位:训练时没见过的词完全没有向量,形态丰富语言里
        "ev" 和 "evler" 完全不相关。FastText 反问:为什么不把词拆成 character n-gram?
        每个词表示为它的 n=3 到 6 的 n-gram 集合,加上特殊边界符 &lt;&gt; 标记词首词尾,
        再加整词 token 保留整词信号。
      </p>

      <NgramDecomposeDiagram word={word} />
      <p className={styles.caption}>
        ↑ 蓝色为字符序列(含边界符),粉色行为各 n 值下滑动产生的 n-gram,绿色为整词 token。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {DEMO_WORDS.map((w) => (
          <button key={w} type="button" onClick={() => setWord(w)} style={btnStyle(word === w)}>{w}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={FASTTEXT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={FASTTEXT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 n ∈ [3, 6]?
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>n=2 太短</strong>:"ap" 在 apple/apricot/cap/map 都出现,几乎无区分度</li>
              <li><strong>n=8+ 太长</strong>:接近词级,失去共享形态部件的优势</li>
              <li><strong>边界符 &lt;&gt;</strong>:让 "her"(词内)和 &lt;her&gt;(独立词)区分,避免语义污染</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
