import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { PREFIX_TUNING_SOURCE_PATH } from "../lib/prose";
import { PARAM_BUDGET_GPT2_LARGE, PARAM_BUDGET_PEFT_EXAMPLE } from "../lib/data";
import { MlpReparamDiagram } from "../widgets/MlpReparamDiagram";
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

export function MlpReparamStage({ mechanism2Prose }: Props) {
  const [example, setExample] = useState<"paper" | "peft">("paper");

  const current =
    example === "paper"
      ? {
          label: `论文参数账:GPT-2 large(354M),m=${PARAM_BUDGET_GPT2_LARGE.m}`,
          trainable: PARAM_BUDGET_GPT2_LARGE.prefixParams,
          total: PARAM_BUDGET_GPT2_LARGE.totalParams,
          pct: PARAM_BUDGET_GPT2_LARGE.prefixParamsPct,
        }
      : {
          label: "HuggingFace peft 实测:PrefixTuningConfig(num_virtual_tokens=20)",
          trainable: PARAM_BUDGET_PEFT_EXAMPLE.trainableParams,
          total: PARAM_BUDGET_PEFT_EXAMPLE.totalParams,
          pct: PARAM_BUDGET_PEFT_EXAMPLE.trainablePct,
        };

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:MLP 重参数化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        直接学全部 layer 的 prefix 矩阵在大模型上不稳定。Prefix Tuning 换一种
        参数化:只学一个小规模 P_small,再用 MLP 把它展开到所有层、所有
        attention head 的 K/V —— 训练时优化 P_small 和 MLP,推理时把展开后的
        P 缓存下来,不再需要 MLP 前向。
      </p>

      <MlpReparamDiagram />
      <p className={styles.caption}>
        ↑ P = MLP(P_small),P_small ∈ R^(m×d_small);MLP 把它扩展为
        num_layers × 2(K+V) × hidden_size 规模的 per-layer prefix。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setExample("paper")} style={btnStyle(example === "paper")}>
          论文参数账(GPT-2 large)
        </button>
        <button type="button" onClick={() => setExample("peft")} style={btnStyle(example === "peft")}>
          HuggingFace peft 实测
        </button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              {current.label}
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`trainable params: ${current.trainable.toLocaleString()}
total params:     ${current.total.toLocaleString()}
trainable %:      ${current.pct}%`}</pre>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 10, lineHeight: 1.6 }}>
              两个数字来源不同(论文估算 vs 库实现细节),但都落在 ~0.02-0.1%
              量级 —— 比 Adapter(~1-3%)再省一个数量级。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
