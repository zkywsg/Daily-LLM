import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEEPSEEK_R1_SOURCE_PATH } from "../lib/prose";
import { MultiStagePipelineDiagram } from "../widgets/MultiStagePipelineDiagram";
import { MULTI_STAGE_PIPELINE } from "../lib/data";
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

export function MultiStageStage({ mechanism3Prose, synergyProse }: Props) {
  const [stageIdx, setStageIdx] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:R1 多阶段训练 — 把 reasoning 内核裹进通用模型
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        R1-Zero 证明了 reasoning 能从 RL 涌现,但有可读性差、通用任务弱两个缺陷。
        R1 用 4 阶段训练补齐:冷启动 SFT 解决可读性,推理 RL 训出 reasoning 内核,
        拒绝采样 SFT 加回通用能力,RLHF 做最终对齐。
      </p>

      <MultiStagePipelineDiagram activeStage={stageIdx} />
      <p className={styles.caption}>
        ↑ 点按钮查看每个阶段的具体作用,绿色高亮的 Stage 2 是 reasoning 能力的真正来源。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        {MULTI_STAGE_PIPELINE.map((s, i) => (
          <button key={i} type="button" onClick={() => setStageIdx(i)} style={btnStyle(stageIdx === i)}>{s.short}</button>
        ))}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 12, lineHeight: 1.6 }}>
        {MULTI_STAGE_PIPELINE[stageIdx].desc}
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,R1 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 GRPO,没有强 base</strong>:RL 无中生有不出来,policy 几乎全是无效探索</li>
              <li><strong>只有强 base + GRPO,没有干净 reward</strong>:PRM 被 reward hacking,模型学会输出"看起来对的废话"</li>
              <li><strong>只有强 base + 干净 reward,没有省内存 GRPO</strong>:PPO + value model 在 7B+ 模型上跑不起来</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
