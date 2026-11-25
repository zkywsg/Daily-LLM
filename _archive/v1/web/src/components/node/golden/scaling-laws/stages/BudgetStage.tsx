import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SCALING_SOURCE_PATH } from "../lib/prose";
import { ComputeCalculator } from "../widgets/ComputeCalculator";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const PRESETS = [
  { label: "10²¹ (LLaMA-1 7B)", log: 21 },
  { label: "10²³ (GPT-3)",       log: 23 },
  { label: "10²⁴ (Chinchilla)",  log: 24 },
  { label: "10²⁵ (GPT-4 估)",   log: 25 },
  { label: "10²⁶ (未来)",        log: 26 },
];

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function BudgetStage({ mechanism3Prose, synergyProse }: Props) {
  const [presetIdx, setPresetIdx] = useState(2);
  const preset = PRESETS[presetIdx];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:从 scaling law 到训练 / 部署预算
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Scaling law 的真正价值在于把 "训练投资决策" 变成可计算的工程问题:
        给定算力预算 C → Chinchilla 公式直接给出 (N, D) 最优。
        但实际部署阶段推理成本主导,LLaMA 路线故意 over-train 小模型,接受训练浪费换推理高效 —
        Chinchilla 是单次训练最优,LLaMA 是部署阶段最优。
      </p>

      <ComputeCalculator flopsLog={preset.log} />
      <p className={styles.caption}>
        ↑ 切换不同算力预算看 Chinchilla 最优 N 和 D · 同时给出多卡训练时长估算。
        从 LLaMA-1 量级到 GPT-4 估计量级跨 4 个数量级。
      </p>

      <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          算力预算预设
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {PRESETS.map((p, i) => (
            <button key={i} type="button" onClick={() => setPresetIdx(i)} style={btnStyle(presetIdx === i)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SCALING_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SCALING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              三件套对决策的影响
            </div>
            <ol style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 18, lineHeight: 1.7, margin: 0 }}>
              <li><strong>经验幂律(Kaplan)</strong>:让"小模型实验外推到大模型"成为预算决策的科学基础</li>
              <li><strong>等比配比(Chinchilla)</strong>:修正 Kaplan 的方向偏差 — N 大 D 小 → N=D 平衡,GPT-3 路线被淘汰</li>
              <li><strong>推理成本视角(LLaMA)</strong>:从训练最优到部署最优 → 开源 LLM 生态可行,7B / 8B 主导消费市场</li>
            </ol>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, fontStyle: "italic", lineHeight: 1.5 }}>
              三者合起来才能回答"我的预算该造多大模型 / 训多久 / 谁能部署得起"
              — 任意一个缺位 LLM 生态都不会是今天的样子。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
