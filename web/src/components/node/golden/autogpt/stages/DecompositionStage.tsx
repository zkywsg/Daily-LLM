import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUTOGPT_SOURCE_PATH } from "../lib/prose";
import { TaskDecompositionDiagram } from "../widgets/TaskDecompositionDiagram";
import { DECOMPOSED_TASKS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function DecompositionStage({ intuitionProse, mechanism1Prose }: Props) {
  const [revealed, setRevealed] = useState(DECOMPOSED_TASKS.length);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Task Decomposition — 给 ReAct 喂"问题"
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ReAct 解决了"给具体问题,LLM 多步答"的循环,但前提是人已经把问题
        想清楚了。给一个模糊高级目标,ReAct 直接懵 — LLM 不知道"研究"
        具体指什么。AutoGPT 在 ReAct 外面再包一层:LLM 自己把目标拆成
        子任务队列,否则 ReAct 根本没有"问题"可循环。
      </p>

      <TaskDecompositionDiagram revealedCount={revealed} />
      <p className={styles.caption}>
        ↑ 拖动逐个揭示 LLM 自动分解出的子任务 — 从"研究市场"这句模糊指令到 5 个可执行的具体任务。
      </p>
      <input
        type="range"
        min={0}
        max={DECOMPOSED_TASKS.length}
        step={1}
        value={revealed}
        onChange={(e) => setRevealed(parseInt(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={AUTOGPT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={AUTOGPT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              与 ReAct 的关键区别
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              人接到模糊目标也是先在脑子里拆任务、做、看效果、再调整。
              AutoGPT 让 LLM 自己分解、自己执行、自己反思、自己重排,
              完全无人干预 — 这是它相对 ReAct 的核心升级。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
