import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT3_SOURCE_PATH } from "../lib/prose";
import { ICL_EXAMPLES } from "../lib/scaling";
import { ICLPromptCompare } from "../widgets/ICLPromptCompare";
import { EmergentAbilityCurve } from "../widgets/EmergentAbilityCurve";
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

export function InContextLearningStage({ mechanism2Prose }: Props) {
  const [exampleIdx, setExampleIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:In-Context Learning — 不更新权重的"伪学习"
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        给模型在 prompt 里塞几个示例,它就能模仿格式完成新输入。这件事不需要
        参数更新 —— GPT-3 的权重在推理时完全冻结。它学的不是任务本身,
        而是"看到这种 prompt 该怎么续"的元能力。
      </p>

      <ICLPromptCompare exampleIdx={exampleIdx} />
      <p className={styles.caption}>
        ↑ 同一任务三种 shot 数,只是 prompt 里多塞几个示例,准确率就能差 30+ 个点。
        这是 GPT-3 论文 Fig 1.2 / 1.3 最震撼的发现。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GPT3_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <EmergentAbilityCurve />
          <p className={styles.caption}>
            "涌现"(emergent ability)指有些能力在参数小时几乎全 0,
            过临界后突然跳到可用 —— 三位数加法、波斯语 QA 这类纯文本推理任务
            都是典型。小模型怎么训也做不到,只能等规模上去。
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
              选 ICL 任务
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {ICL_EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setExampleIdx(i)}
                  style={btnStyle(i === exampleIdx)}
                >
                  {ex.task} — {ex.description}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
