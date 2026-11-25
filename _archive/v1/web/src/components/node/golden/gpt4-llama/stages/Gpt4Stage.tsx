import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT4_LLAMA_SOURCE_PATH } from "../lib/prose";
import { DualTrackDiagram } from "../widgets/DualTrackDiagram";
import { ContextWindowGrowthChart } from "../widgets/ContextWindowGrowthChart";
import { CONTEXT_GROWTH } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function Gpt4Stage({ intuitionProse, mechanism1Prose }: Props) {
  const [visibleSteps, setVisibleSteps] = useState(CONTEXT_GROWTH.length);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:GPT-4 — 闭源前沿的形态(MoE + 多模态 + 可预测 scaling)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GPT-4 的技术细节几乎完全不公开,但行业共识是 16 专家 MoE、~1.8T 参数、
        ~13T 训练 token。三个特有技术:MoE 架构首次大规模用于前沿 LLM、
        多模态从预训练阶段原生训练(而非事后桥接)、以及用 10000× 小算力的
        loss 就能预测最终性能——scaling law 在万亿参数级仍精确成立。
      </p>

      <DualTrackDiagram />
      <p className={styles.caption}>
        ↑ GPT-4 的估计规格一览(技术报告不透露细节,来自行业推测)。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <ContextWindowGrowthChart visibleSteps={visibleSteps} />
        <p className={styles.caption}>
          ↑ 拖动看上下文窗口如何从 8K 扩展到 128K。
        </p>
        <input
          type="range"
          min={1}
          max={CONTEXT_GROWTH.length}
          step={1}
          value={visibleSteps}
          onChange={(e) => setVisibleSteps(parseInt(e.target.value))}
          style={{ width: "100%", marginTop: 8 }}
        />
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              GPT-4 特有的三项技术
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>MoE</strong> — 16 专家×110B,每次仅激活 2 个(~220B),总参数 1.76T</li>
              <li><strong>多模态原生预训练</strong> — vision encoder + LLM 端到端训练,非事后桥接</li>
              <li><strong>可预测 scaling</strong> — 10000× 小算力 loss 预测最终性能,误差 &lt;5%</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
