import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ALBERT_SOURCE_PATH } from "../lib/prose";
import { FactorizedEmbeddingDiagram } from "../widgets/FactorizedEmbeddingDiagram";
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

const E_PRESETS = [64, 128, 256, 768];

export function FactorizedEmbeddingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [embedSize, setEmbedSize] = useState(128);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Embedding 因式分解
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        BERT 的 token embedding 矩阵是 V × H(vocab_size × hidden_size)— token
        本身只承载词汇信息,不需要和 hidden state 一样宽。ALBERT 把 embedding
        拆成两段:先投影到小的 E 维空间(V × E),再升维到 H(E × H),
        E 典型取 128,比 H 小 6–32 倍。
      </p>

      <FactorizedEmbeddingDiagram embedSize={embedSize} />
      <p className={styles.caption}>
        ↑ 拖动 / 切换 E 的取值,看 V×E + E×H 相对 V×H 的参数压缩效果。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {E_PRESETS.map((e) => (
          <button key={e} type="button" onClick={() => setEmbedSize(e)} style={btnStyle(embedSize === e)}>
            E = {e}
          </button>
        ))}
      </div>
      <input
        type="range"
        min={32}
        max={768}
        step={16}
        value={embedSize}
        onChange={(evt) => setEmbedSize(Number(evt.target.value))}
        style={{ width: "100%", marginTop: 12 }}
        aria-label="Embedding 维度 E 滑块"
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={ALBERT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={ALBERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 E 可以远小于 H?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              Token embedding 的"信息容量"由 vocab_size 决定 —— 30K 个词的查表,
              不需要随 hidden_size 线性增长。hidden state 才需要承载上下文相关的
              深层表征,必须维持较大维度 H。强制两者维度相等,是 BERT 里最直接的浪费。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
