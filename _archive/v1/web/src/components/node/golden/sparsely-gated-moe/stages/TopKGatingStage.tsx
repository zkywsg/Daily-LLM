import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SPARSELY_GATED_MOE_SOURCE_PATH } from "../lib/prose";
import { DEMO_TOKENS } from "../lib/data";
import { TopKRoutingDiagram } from "../widgets/TopKRoutingDiagram";
import { ParamsVsComputeChart } from "../widgets/ParamsVsComputeChart";
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

export function TopKGatingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [sentenceIdx, setSentenceIdx] = useState(0);
  const [tokenIdx, setTokenIdx] = useState(0);
  const sentence = DEMO_TOKENS[sentenceIdx];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Top-K Sparse Gating — 让稀疏可微
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        gating 网络把每个 token 投影到 N=2048 个 expert 的 logits,noisy top-K
        只保留 4 个最大值(其余设为 -∞),softmax 后精确 0 —— 未选中的 2044 个
        expert 这一步完全不计算。这是 sparse gating 与 dense gating 的根本区别。
      </p>

      <ParamsVsComputeChart />
      <p className={styles.caption}>
        ↑ 137B 总参数,每 token 只激活 1.5B —— 参数与算力首次解耦。
        dense 模型(LSTM)两条柱重合,MoE 的总参数条越堆越高但激活条几乎不变。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <TopKRoutingDiagram sentenceIdx={sentenceIdx} tokenIdx={tokenIdx} />
          <p className={styles.caption}>
            选一个 token,看 gating 从 N=2048 个 expert(下方展示 64 个采样)里
            选出 top-K=4 个(粉色 cell + softmax 权重)。灰色 cell 精确为 0,
            不参与这一步计算。
          </p>
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              示例句子
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: "var(--space-3)" }}>
              {DEMO_TOKENS.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => {
                    setSentenceIdx(i);
                    setTokenIdx(0);
                  }}
                  style={btnStyle(i === sentenceIdx)}
                >
                  {s.label}: {s.tokens.join(" ")}
                </button>
              ))}
            </div>
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              选一个 token
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {sentence.tokens.map((t, i) => (
                <button key={i} type="button" onClick={() => setTokenIdx(i)} style={btnStyle(i === tokenIdx)}>
                  {t}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
