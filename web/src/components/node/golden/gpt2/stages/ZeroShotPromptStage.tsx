import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT2_SOURCE_PATH } from "../lib/prose";
import { PromptTaskCard } from "../widgets/PromptTaskCard";
import { ZeroShotVsSftBars } from "../widgets/ZeroShotVsSftBars";
import { PROMPT_TEMPLATES } from "../lib/data";
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

export function ZeroShotPromptStage({ mechanism2Prose }: Props) {
  const [taskIdx, setTaskIdx] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Zero-Shot Prompt — 自然语言替代 task head
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        BERT / GPT-1 时代:每个任务接 task head + 监督微调。GPT-2 反问:
        如果 LM 在预训练里已经隐式见过所有 "task → answer" 模式,
        是否只需要 prompt 就能激活?把任务从 "训专用模型" 压成 "对 LM 写 prompt 让它续写"。
      </p>

      <PromptTaskCard taskIdx={taskIdx} />
      <p className={styles.caption}>
        ↑ 不同 prompt 模板触发不同任务,所有任务共享同一个 GPT-2 权重,
        权重一字节都不动。点下方按钮看 4 种典型 prompt 模式。
      </p>

      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {PROMPT_TEMPLATES.map((t, i) => (
          <button key={i} type="button" onClick={() => setTaskIdx(i)} style={btnStyle(taskIdx === i)}>{t.task}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GPT2_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ZeroShotVsSftBars />
          <p className={styles.caption}>
            ↑ 论文 Table 3。GPT-2 XL zero-shot(粉)对前作 zero-shot(灰)全面领先;
            CBT-NE 上 89.1% 已经超过监督 SOTA 87.7%。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4, fontWeight: 600, textTransform: "uppercase" }}>关键观察</div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>翻译用 prompt <code>Translate to French:</code> 直接 work — 没接任何 encoder-decoder</li>
              <li>摘要用 <code>TL;DR:</code> 触发 — WebText 里 Reddit / 论坛见过无数次这个模式</li>
              <li>QA 用 <code>Q: ... A:</code> 触发 — StackOverflow / Quora 见过的格式</li>
              <li>所有 zero-shot 能力都来自 "预训练时见过的模式 + 大容量记忆"</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
