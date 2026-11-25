import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { INSTRUCTGPT_SOURCE_PATH } from "../lib/prose";
import { SFT_DEMOS } from "../lib/data";
import { RLHFPipelineFlow } from "../widgets/RLHFPipelineFlow";
import { SFTBeforeAfter } from "../widgets/SFTBeforeAfter";
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

export function SFTStage({ intuitionProse, mechanism1Prose }: Props) {
  const [demoIdx, setDemoIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:SFT — 用人类示范热启动
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        GPT-3 不是\"不会\"完成任务,而是\"不知道你想要什么\" —— 它的训练目标是
        续写 web 文本,看到 \"解释机会成本\" 这种指令时,它的本能是续写
        \"…provide a real-world example…\" 这种 stack overflow 风格的提示词文档。
        SFT 用 13K labeler 示范的 prompt-response 对 fine-tune,把模型切换
        到\"听懂指令\"模式。
      </p>

      <RLHFPipelineFlow activeStage={0} />
      <p className={styles.caption}>
        ↑ RLHF 三阶段总览,黄框 SFT 是起点。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SFTBeforeAfter demoIdx={demoIdx} />
          <p className={styles.caption}>
            切换示范看 \"对齐\" 的本质 —— 不是更多知识,是\"听懂指令\"。
            注意 GPT-3 不是不会,而是把指令当成 prompt 文档接着续写。
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
              示范例子
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {SFT_DEMOS.map((d, i) => (
                <button key={i} type="button" onClick={() => setDemoIdx(i)} style={btnStyle(i === demoIdx)}>
                  {d.prompt}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
