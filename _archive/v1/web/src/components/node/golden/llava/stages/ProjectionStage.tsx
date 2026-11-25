import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LLAVA_SOURCE_PATH } from "../lib/prose";
import { ProjectionDiagram } from "../widgets/ProjectionDiagram";
import { BridgeParamCompareChart } from "../widgets/BridgeParamCompareChart";
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

export function ProjectionStage({ mechanism2Prose }: Props) {
  const [mode, setMode] = useState<"linear" | "mlp">("linear");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Linear / MLP Projection — 把 vision feature 投到 LLM 空间
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        CLIP feature 是 768 维,LLaMA-7B 的 token embedding 是 4096 维,
        两者不在同一空间。LLaVA 加一个单层 linear projection(LLaVA-1.5
        升级到 2 层 MLP)把 768 维投到 LLM 空间。投影后 visual token 就和
        文本 token "看起来一样",直接拼到 LLM input 序列里,LLM 一行架构
        改动都不需要。
      </p>

      <ProjectionDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看单 Linear(LLaVA-1.0)vs 2 层 MLP(LLaVA-1.5)的投影结构差异。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("linear")} style={btnStyle(mode === "linear")}>单 Linear(1.0)</button>
        <button type="button" onClick={() => setMode("mlp")} style={btnStyle(mode === "mlp")}>2 层 MLP(1.5)</button>
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <BridgeParamCompareChart />
        <p className={styles.caption}>
          ↑ LLaVA 的桥接模块比 BLIP-2 Q-Former 小 47×,比 Flamingo Perceiver Resampler 小 100× — 简洁有效胜过复杂精巧。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={LLAVA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              这是 LLaVA 唯一从零训练的组件
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              参数账:projection 只有 768×4096 ≈ 3M(Linear)或 ~30M(2 层
              MLP),相比 LLaMA-7B 的 7B 几乎可忽略。比起 BLIP-2 的
              Q-Former(188M,需要复杂训练)或 Flamingo 的 Perceiver
              Resampler(数百 M),LLaVA 的极简 projection 是质的简化。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
