import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SPARSELY_GATED_MOE_SOURCE_PATH } from "../lib/prose";
import { AuxLossBalancingDiagram } from "../widgets/AuxLossBalancingDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function AuxLossStage({ mechanism2Prose }: Props) {
  const [balanced, setBalanced] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Auxiliary Loss — 防 expert 塌缩
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        不加约束时 gating 会很快塌缩到少数几个 expert,其余永远学不到东西——
        137B 参数实际只在用"热 expert"的一小部分。Importance Loss(权重均衡)
        + Load Loss(token 数均衡)两个 auxiliary loss,首次把这个问题系统性解决,
        后来成为所有 MoE 工作的标配。
      </p>

      <AuxLossBalancingDiagram balanced={balanced} />
      <p className={styles.caption}>
        ↑ 切换 toggle 看负载分布。不加 aux loss:少数 expert 被反复选中,大多数
        闲置;加 aux loss:采样的 16 个 expert 接近均匀(代表 2048 个整体的行为)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div
            style={{
              padding: "var(--space-4)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
              fontSize: "var(--fs-sm)",
              lineHeight: 1.7,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Importance Loss</div>
            <div style={{ color: "var(--ink-secondary)", marginBottom: 12 }}>
              L = w · CV(Importance_i)²,Importance_i = Σ G(x)_i —— 让每个
              expert 被选中的总 gate 权重均衡。
            </div>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Load Loss</div>
            <div style={{ color: "var(--ink-secondary)" }}>
              L = w · CV(Load_i)² —— 让每个 expert 被路由到的 token 数均衡,
              防止某 expert 被过度激活(容量溢出)。
            </div>
          </div>
          <p className={styles.caption}>
            CV 是变异系数(std / mean),小 = 均衡。总 loss = task loss +
            importance loss + load loss —— 这个 auxiliary loss 框架是
            Switch / DeepSpeed-MoE / DeepSeek-V3 所有 load balancing 方案的祖先。
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
              是否加 aux loss
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { v: false, label: "不加 → 塌缩" },
                { v: true, label: "加 → 均衡" },
              ].map((opt) => {
                const active = balanced === opt.v;
                return (
                  <button
                    key={String(opt.v)}
                    type="button"
                    onClick={() => setBalanced(opt.v)}
                    style={{
                      padding: "4px 12px",
                      fontSize: "var(--fs-sm)",
                      borderRadius: "var(--radius-sm)",
                      border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
                      background: active ? "var(--accent-link)" : "var(--bg-surface)",
                      color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
