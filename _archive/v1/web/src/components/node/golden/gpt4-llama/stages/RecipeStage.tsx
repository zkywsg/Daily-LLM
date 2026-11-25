import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT4_LLAMA_SOURCE_PATH } from "../lib/prose";
import { RecipeEvolutionDiagram } from "../widgets/RecipeEvolutionDiagram";
import { RecipeAdoptionDiagram } from "../widgets/RecipeAdoptionDiagram";
import { RECIPE_TABLE } from "../lib/data";
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

export function RecipeStage({ mechanism3Prose, synergyProse }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:现代 LLM 配方 — LLaMA 的 6 件套替换
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        LLaMA-1 的架构是当时所有"现代 LLM 改进"的集大成:Pre-LN → RMSNorm、
        正余弦/learned PE → RoPE、ReLU/GELU → SwiGLU、Multi-Head → GQA、
        BPE → SentencePiece。这套配方在 LLaMA-2/3、Mistral、Qwen、Yi、
        DeepSeek 上几乎完全一致,只有数据组成 / 训练策略 / 后训练有差异。
      </p>

      <RecipeEvolutionDiagram highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 点按钮聚焦某个组件,查看 2017 → GPT-3 → LLaMA 的演化路径。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {RECIPE_TABLE.map((r, i) => (
          <button key={r.component} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{r.component}</button>
        ))}
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <RecipeAdoptionDiagram />
        <p className={styles.caption}>
          ↑ 2023 年之后几乎所有主流开源 LLM 都采用这套现代配方。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,2023 转折点都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 GPT-4 闭源</strong>:生态停留在几家闭源公司,没有开放底座做微调/对齐研究</li>
              <li><strong>只有 LLaMA 开源</strong>:社区不知道 scaling 是否还能走,前沿停止推进</li>
              <li><strong>只有两条路线,没有配方定型</strong>:各家架构不同,推理框架难以通用</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
