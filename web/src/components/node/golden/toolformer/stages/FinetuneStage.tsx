import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TOOLFORMER_SOURCE_PATH } from "../lib/prose";
import { ToolformerVsGpt3Chart } from "../widgets/ToolformerVsGpt3Chart";
import { BENCHMARK_TABLE } from "../lib/data";
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

export function FinetuneStage({ mechanism3Prose, synergyProse }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:微调内化 — 把 tool use 写进参数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把过滤后的 (text with API call) 数据混合原 pretraining 语料,微调
        LM。模型学到:在合适的时机自动插入 [Tool(args)] token,然后用
        返回结果继续生成。推理时不再需要任何 prompt 提示 — 调工具成了
        模型的默认行为。
      </p>

      <ToolformerVsGpt3Chart highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 点按钮聚焦某个任务 — Toolformer 6.7B 在数学/QA 任务上大幅反超参数量 25× 的 GPT-3 175B。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {BENCHMARK_TABLE.map((r, i) => (
          <button key={r.task} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{r.task.split("(")[0]}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={TOOLFORMER_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={TOOLFORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,反超不会发生
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有自采样</strong>:候选海量但质量参差,微调后模型学到"乱调工具"</li>
              <li><strong>只有 Perplexity 过滤</strong>:没有自采样提供大批候选,失去自监督优势</li>
              <li><strong>只有微调</strong>:没有过滤,无用调用喂进去,工具能力没学到反而拖累流畅性</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
