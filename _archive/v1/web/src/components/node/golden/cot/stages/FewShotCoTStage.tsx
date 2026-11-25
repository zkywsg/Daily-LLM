import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { COT_SOURCE_PATH } from "../lib/prose";
import { QA_EXAMPLES } from "../lib/data";
import { StandardVsCoTCompare } from "../widgets/StandardVsCoTCompare";
import { StepLighting } from "../widgets/StepLighting";
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

export function FewShotCoTStage({ intuitionProse, mechanism1Prose }: Props) {
  const [exampleIdx, setExampleIdx] = useState(1);
  const [steps, setSteps] = useState(2);

  const ex = QA_EXAMPLES[exampleIdx];
  const totalSteps = ex.cotSteps.length + 1;

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Few-shot CoT — 示例驱动的思维链
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        在 prompt 里给模型 1-2 个"问题 + 推理过程 + 答案"的完整示例。
        模型看到格式后,会模仿着把自己的推理也显式写出来 —— 这把模型从
        "直觉式猜答案"切换到"逐步推导",数学/逻辑/常识题准确率显著上升。
      </p>

      <StandardVsCoTCompare exampleIdx={exampleIdx} />
      <p className={styles.caption}>
        ↑ 同一道题,Standard 直接给答案常错;CoT 显式拆步反而做对。
        切换题目看不同例子里的对比。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={COT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={COT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <StepLighting exampleIdx={exampleIdx} visibleSteps={steps} />
          <p className={styles.caption}>
            拖 step slider 模拟模型"一步一步想"的过程。每点亮一步,
            前面所有步骤的推理在 prompt context 里都已经存在 —— 模型在自己的
            生成里建立中间结果。
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
              示例题
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: "var(--space-3)" }}>
              {QA_EXAMPLES.map((q, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => {
                    setExampleIdx(i);
                    setSteps(0);
                  }}
                  style={btnStyle(i === exampleIdx)}
                >
                  题 {i + 1}: {q.question.slice(0, 30)}…
                </button>
              ))}
            </div>
            <label
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                marginBottom: 4,
              }}
            >
              <span>已显示步骤</span>
              <strong>{steps} / {totalSteps}</strong>
            </label>
            <input
              type="range"
              min={0}
              max={totalSteps}
              step={1}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10))}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
