import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT1_SOURCE_PATH } from "../lib/prose";
import { TaskFormatCard } from "../widgets/TaskFormatCard";
import { AuxLossDemo } from "../widgets/AuxLossDemo";
import { TASK_FORMATS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function UnifiedInterfaceStage({ mechanism3Prose, synergyProse }: Props) {
  const [taskIdx, setTaskIdx] = useState(1);
  const [lambda, setLambda] = useState(0.5);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:统一任务接口 — 所有任务转成序列输入
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GPT-1 的核心工程贡献:用 <code>&lt;s&gt; / &lt;$&gt; / &lt;/s&gt;</code> 三个特殊 token
        把 4 类任务(分类/NLI/相似度/多选)转成统一序列格式,模型本身没有任务特定模块,
        只多一个 linear head。微调加 λ·L_LM auxiliary loss 防止遗忘预训练学到的通用语言能力。
      </p>

      <TaskFormatCard taskIdx={taskIdx} />
      <p className={styles.caption}>
        ↑ 切换看 4 种任务格式如何统一表达。多选任务要 N 次 forward(每个候选一次)后 softmax。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {TASK_FORMATS.map((t, i) => (
          <button key={i} type="button" onClick={() => setTaskIdx(i)} style={btnStyle(taskIdx === i)}>{t.name}</button>
        ))}
      </div>

      <AuxLossDemo lambda={lambda} />
      <p className={styles.caption}>
        ↑ λ 权衡任务学习速度 vs 通用语言能力保留。论文用 λ=0.5,是防 catastrophic forgetting 的早期工程解。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>λ</span><strong>{lambda.toFixed(2)}</strong>
      </label>
      <input type="range" min={0} max={0.9} step={0.05} value={lambda}
             onChange={(e) => setLambda(parseFloat(e.target.value))} style={{ width: "100%" }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GPT1_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GPT1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              统一接口的种子
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              GPT-1 的"所有任务转成序列输入"思想直接预示了 GPT-3 的 in-context learning —
              任务描述 + 例子全部用自然语言写在 prompt 里,<strong>连 task-specific head 都不要</strong>。
              从这角度,GPT-3 就是 GPT-1 接口的极端化。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
