import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DPO_SOURCE_PATH } from "../lib/prose";
import { PpoVsDpoPipeline } from "../widgets/PpoVsDpoPipeline";
import { MethodComplexityBars } from "../widgets/MethodComplexityBars";
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

export function PipelineStage({ intuitionProse, mechanism1Prose }: Props) {
  const [side, setSide] = useState<"ppo" | "dpo" | "both">("both");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:从 PPO 反推 — 最优 policy 有 closed-form 解
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        RLHF 的目标 max E[r] − β · KL 在固定 r 和 π_ref 下有闭式最优解 π*(y|x) ∝ π_ref · exp(r/β)。
        反过来 reward 也可以用 policy 表达:r = β · log(π/π_ref) + β · log Z(x)。
        这意味着 LLM 本身就是个 reward model — 不需要单独训 RM。
      </p>

      <PpoVsDpoPipeline highlight={side} />
      <p className={styles.caption}>
        ↑ 同一份偏好数据,PPO 三阶段 (SFT→RM→PPO) vs DPO 一阶段。
        DPO 跳过 RM 和 RL,直接用 cross-entropy 形式 loss 训 actor。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("ppo")} style={btnStyle(side === "ppo")}>聚焦 PPO</button>
        <button type="button" onClick={() => setSide("dpo")} style={btnStyle(side === "dpo")}>聚焦 DPO</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DPO_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DPO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <MethodComplexityBars />
          <p className={styles.caption}>
            ↑ 4 维度对比。代码行数用 log 比例 — 实际差 17×。
            训练成本约 PPO 的 1/20。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              数学关键奇迹
            </div>
            <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", lineHeight: 1.6 }}>
              <p style={{ margin: "0 0 6px" }}>
                Z(x) 是难算的归一化 — 需要对整个生成空间求和。
              </p>
              <p style={{ margin: "0 0 6px" }}>
                但在 BT 偏好的 log-ratio 相减时,两个 Z(x) 抵消(同一个 x)。
              </p>
              <p style={{ margin: 0, fontWeight: 600 }}>
                → 整个 loss 变成可直接对 (chosen, rejected) 求梯度的 cross-entropy。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
