import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUTOGPT_SOURCE_PATH } from "../lib/prose";
import { CritiqueReplanDiagram } from "../widgets/CritiqueReplanDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
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

export function CritiqueStage({ mechanism3Prose, synergyProse }: Props) {
  const [showAfter, setShowAfter] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Self-Critique + Replan — 防止卡死或走偏
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        每步执行后让 LLM 评估:"这一步做得对吗?需要重做吗?子任务列表要
        调整吗?"这是 Reflexion 正式化的能力,AutoGPT 早期版本已经在用。
        没有这一步,LLM 容易卡死在某个子任务里循环失败,或者一路走偏
        而不自知。
      </p>

      <CritiqueReplanDiagram showAfter={showAfter} />
      <p className={styles.caption}>
        ↑ 切换看反思前后 task queue 的变化 — LLM 主动插入验证步骤,而不是机械执行原计划。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setShowAfter(false)} style={btnStyle(!showAfter)}>反思前</button>
        <button type="button" onClick={() => setShowAfter(true)} style={btnStyle(showAfter)}>反思后</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={AUTOGPT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={AUTOGPT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,自主 agent 都跑不起来
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 Task Decomposition</strong>:子任务列出来了,但没循环/没记忆,一个任务跑完就停</li>
              <li><strong>只有 Tool Loop + Memory</strong>:能长跑,但没分解,给个模糊目标直接懵</li>
              <li><strong>只有 Self-Critique</strong>:能反思,但没分解也没记忆,反思的对象不存在</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
