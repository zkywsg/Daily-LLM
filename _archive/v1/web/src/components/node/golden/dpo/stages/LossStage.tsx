import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DPO_SOURCE_PATH } from "../lib/prose";
import { DpoLossDataflow } from "../widgets/DpoLossDataflow";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

type Highlight = "actor" | "ref" | "logratio" | "loss" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function LossStage({ mechanism2Prose }: Props) {
  const [hl, setHl] = useState<Highlight>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Bradley-Terry + 隐 reward → DPO loss
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        InstructGPT 的 RM 训练用 BT 偏好模型 P(y_w ≻ y_l | x) = σ(r_w − r_l)。
        把机制一里 reward = β · log(π/π_ref) 代回来,两个 Z(x) 抵消,
        得到一个 cross-entropy 形式的损失 — 完全不用 RM 也不用 PPO,
        actor + π_ref 两个模型就够。
      </p>

      <DpoLossDataflow highlight={hl} />
      <p className={styles.caption}>
        ↑ 一条 (prompt, y_w, y_l) → actor 算 2 个 log-prob + ref 算 2 个 log-prob
        → log-ratio 相减 → σ → −log 得到 loss。点按钮聚焦各阶段。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("actor")} style={btnStyle(hl === "actor")}>actor π_θ</button>
        <button type="button" onClick={() => setHl("ref")} style={btnStyle(hl === "ref")}>π_ref (冻结)</button>
        <button type="button" onClick={() => setHl("logratio")} style={btnStyle(hl === "logratio")}>log-ratio</button>
        <button type="button" onClick={() => setHl("loss")} style={btnStyle(hl === "loss")}>−log σ</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DPO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              DPO 训练循环代码 — 总共就这么多
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`# Actor on chosen/rejected
a_w = log_prob(actor, prompts, chosen)
a_l = log_prob(actor, prompts, rejected)

# Frozen reference baseline
with torch.no_grad():
    r_w = log_prob(ref, prompts, chosen)
    r_l = log_prob(ref, prompts, rejected)

# DPO loss
chosen_logratio   = a_w - r_w
rejected_logratio = a_l - r_l
logits = beta * (chosen_logratio - rejected_logratio)
loss = -F.logsigmoid(logits).mean()

loss.backward()  # done`}</pre>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              对比 PPOTrainer
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              TRL <code>DPOTrainer</code> 几百行;<code>PPOTrainer</code> 几千行;
              工程复杂度差一个数量级 — 这是 DPO 民主化 RLHF 的真正原因。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
