import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { O1_SOURCE_PATH } from "../lib/prose";
import { TestTimeScalingChart } from "../widgets/TestTimeScalingChart";
import { COMPUTE_TRADEOFF } from "../lib/data";
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

export function TestTimeScalingStage({ mechanism2Prose }: Props) {
  const [task, setTask] = useState("all");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Test-Time Compute Scaling — 推理时算力也是新 scaling 轴
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        在固定模型上,thinking token 数翻倍,准确率持续上升。这条 log-log 曲线斜率
        本身就是一条"test-time scaling 指数"— 把 Kaplan / Chinchilla scaling laws
        从训练阶段扩展到了推理阶段。推理算力的边际性价比在 reasoning 任务上远高于
        训练算力,这直接改变了 LLM 商业模型。
      </p>

      <TestTimeScalingChart activeTask={task} />
      <p className={styles.caption}>
        ↑ AIME / Codeforces / GPQA 三个任务的 thinking tokens vs 准确率曲线(log 尺度),点按钮聚焦某个任务。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setTask("all")} style={btnStyle(task === "all")}>全部</button>
        {["AIME", "Codeforces", "GPQA"].map((t) => (
          <button key={t} type="button" onClick={() => setTask(t)} style={btnStyle(task === t)}>{t}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={O1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              训练算力 vs 推理算力
            </div>
            {COMPUTE_TRADEOFF.map((row) => (
              <div key={row.axis} style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 8, lineHeight: 1.5 }}>
                <strong>{row.axis}</strong>:算力 {row.multiplier} → 准确率 {row.accuracyGain}
              </div>
            ))}
            <p style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              o1-mini → o1 → o1-pro($200/月) → o3($1000/query),开发者第一次能"按问题难度选算力"。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
