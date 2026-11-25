import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUTOGPT_SOURCE_PATH } from "../lib/prose";
import { ToolLoopMemoryDiagram } from "../widgets/ToolLoopMemoryDiagram";
import { TOOL_LOOP_STEPS, TOOL_SET } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ToolLoopStage({ mechanism2Prose }: Props) {
  const [step, setStep] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Tool Loop + Persistent Memory — 长 trajectory 不爆 token
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        对每个子任务执行 ReAct 风格 thought-action-observation。但 ReAct
        一次会话所有上下文都在 prompt 里,token 很快爆。AutoGPT 引入
        persistent memory — 把执行历史存到 vector DB,需要时检索相关
        记忆喂回 prompt,而不是全部塞进 context。这让 agent 能跑几小时、
        上千步。
      </p>

      <ToolLoopMemoryDiagram step={step} />
      <p className={styles.caption}>
        ↑ 拖动查看每一步 task → tool → observation → memory 的完整循环,注意 memory 条数逐步累积。
      </p>
      <input
        type="range"
        min={0}
        max={TOOL_LOOP_STEPS.length - 1}
        step={1}
        value={step}
        onChange={(e) => setStep(parseInt(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={AUTOGPT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              默认工具集
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              {TOOL_SET.map((t) => (
                <li key={t}><code>{t}</code></li>
              ))}
            </ul>
            <p style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              可通过 plugin 加新工具(Wolfram Alpha, Twitter API, ...)。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
