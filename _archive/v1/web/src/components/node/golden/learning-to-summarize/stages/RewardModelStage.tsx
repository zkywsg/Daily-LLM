import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LEARNING_TO_SUMMARIZE_SOURCE_PATH } from "../lib/prose";
import { RewardModelTrainingDiagram } from "../widgets/RewardModelTrainingDiagram";
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

export function RewardModelStage({ mechanism2Prose }: Props) {
  const [trained, setTrained] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Bradley-Terry Reward Model — 把偏好编成标量
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        64K 对偏好比较本身只是"A 比 B 好"的离散标签,RL 需要一个连续的标量 reward
        才能算梯度。Reward model 用 Bradley-Terry 负对数似然
        L_RM = −log σ(r(y_w) − r(y_l)) 训练:让 winner 的分数明显高于 loser。
        RM 初始化为 SFT 权重 + 一个 scalar head,只需学"哪个更好"的映射,数据效率高。
      </p>

      <RewardModelTrainingDiagram trained={trained} />
      <p className={styles.caption}>
        ↑ 切换看训练前(分数几乎重叠,loss 很大)vs 训练后(winner/loser 分数明显拉开)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setTrained(false)} style={btnStyle(!trained)}>训练前(随机初始化)</button>
        <button type="button" onClick={() => setTrained(true)} style={btnStyle(trained)}>训练后(收敛)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              Bradley-Terry loss
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>
{`L_RM = -E[log σ(r(y_w) - r(y_l))]

y_w = winner(被偏好)
y_l = loser(被拒绝)
σ  = sigmoid`}
            </pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              等价于二分类:p(win &gt; lose) = σ(r_win − r_lose)。RM 和 SFT 模型同规模(1.3B / 6.7B),1 epoch,lr 1e-5,batch 64。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
