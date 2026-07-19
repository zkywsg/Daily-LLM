import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SWITCH_TRANSFORMER_SOURCE_PATH } from "../lib/prose";
import { DEMO_SENTENCES } from "../lib/data";
import { Top1RoutingDiagram } from "../widgets/Top1RoutingDiagram";
import { RoutingCostCompare } from "../widgets/RoutingCostCompare";
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

export function Top1RoutingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [sentenceIdx, setSentenceIdx] = useState(0);
  const [k, setK] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Top-1 Gating — 每 token 只走一个 expert
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Shazeer 2017 用 top-K(K=4)证明稀疏 MoE 在 LSTM 上 work,但 K=4 意味着
        每个 token 要算 4 个 expert、通信量也是 4 倍。Switch Transformer 的
        关键简化:只要 expert 数量够多(N 最高到 2048),
        <strong>每 token 只路由到 1 个 expert 就够了</strong>
        ——路由计算量和 all-to-all 通信量直接减半到 1/4。
      </p>

      <Top1RoutingDiagram sentenceIdx={sentenceIdx} k={k} />
      <p className={styles.caption}>
        ↑ 切换 top-k 看同一句话的路由行为。k=1 时每行只有 1 个黄色 cell ——
        这就是 Switch Transformer 的核心简化。
      </p>

      <RoutingCostCompare selectedK={k} />
      <p className={styles.caption}>
        ↑ top-1 (Switch) / top-2 (Mixtral) / top-4 (Shazeer 2017) 的算力与通信成本对比。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div
            style={{
              padding: "var(--space-4)",
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
            <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 14 }}>
              {DEMO_SENTENCES.map((s, i) => (
                <button key={i} type="button" onClick={() => setSentenceIdx(i)} style={btnStyle(i === sentenceIdx)}>
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
              top-k 选项
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {[1, 2, 4].map((opt) => (
                <button key={opt} type="button" onClick={() => setK(opt)} style={btnStyle(k === opt)}>
                  top-{opt}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
