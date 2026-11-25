import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MIXTRAL_SOURCE_PATH } from "../lib/prose";
import { DEMO_SENTENCES } from "../lib/data";
import { MoeLayerFlow } from "../widgets/MoeLayerFlow";
import { TokenRoutingDemo } from "../widgets/TokenRoutingDemo";
import { ParamVsActivationBar } from "../widgets/ParamVsActivationBar";
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

export function SparseMoEStage({ intuitionProse, mechanism1Prose }: Props) {
  const [sentenceIdx, setSentenceIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Sparse MoE Layer — router 把 token 路由到 top-k expert
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        把 transformer block 里的 FFN 复制 N 份(N 个 expert),前面加一个
        router 决定每个 token 选哪 k 个 expert。Mixtral 8×7B 是 8 expert + k=2:
        每 token 只跑 2 个 expert,但模型总参数依然容纳 8 份 FFN —— 容量大、算力小。
      </p>

      <MoeLayerFlow />
      <p className={styles.caption}>
        ↑ 单层 MoE 数据流:token → router → top-2 expert 加权求和。
        未选中的 6 个 expert 这一步完全不参与运算 —— 这是\"稀疏激活\"的本质。
      </p>

      <ParamVsActivationBar />
      <p className={styles.caption}>
        三条横向条对比 Mixtral 的参数账:47B 总参数 = Llama-13B 算力 = 70B 质量。
        这是 MoE 工业上能 work 的核心买卖。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={MIXTRAL_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={MIXTRAL_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <TokenRoutingDemo sentenceIdx={sentenceIdx} k={2} />
          <p className={styles.caption}>
            一句话的每个 token 各自选 top-2 expert(粉色 cell + 权重)。
            注意不同 token 路由到不同 expert —— 这是 \"per-token sparse\" 的行为。
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
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {DEMO_SENTENCES.map((s, i) => (
                <button key={i} type="button" onClick={() => setSentenceIdx(i)} style={btnStyle(i === sentenceIdx)}>
                  {s.label}: {s.tokens.join(" ")}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
