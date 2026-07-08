import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LLAVA_SOURCE_PATH } from "../lib/prose";
import { TwoStageTrainingDiagram } from "../widgets/TwoStageTrainingDiagram";
import { TWO_STAGE_TRAINING } from "../lib/data";
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

export function InstructionTuningStage({ mechanism3Prose, synergyProse }: Props) {
  const [stage, setStage] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:两阶段 instruction tuning — feature alignment + visual instruction
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        LLaVA 的训练分两阶段:Stage 1 只训 projection,冻结 CLIP + LLM,用
        CC3M 595K 图文对学会基本对齐;Stage 2 训 projection + LLM,用
        GPT-4 生成的 158K 多模态指令,让 LLaMA 学会"按视觉指令完成任务"
        而不只是描述图像。GPT-4 不需要看图 — 看 caption + bbox 就能生成
        "如果它看了图会怎么回答"的对话。
      </p>

      <TwoStageTrainingDiagram activeStage={stage} />
      <p className={styles.caption}>
        ↑ 点按钮切换看两阶段各自的数据、可训练范围和算力消耗。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        {TWO_STAGE_TRAINING.map((s, i) => (
          <button key={s.stage} type="button" onClick={() => setStage(i)} style={btnStyle(stage === i)}>Stage {i + 1}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={LLAVA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={LLAVA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,LLaVA 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 projection + LLM,没有 CLIP 预对齐</strong>:从零训 vision encoder 算力爆炸</li>
              <li><strong>只有 CLIP + LLM,没有 projection 桥</strong>:两者不在同一坐标系,LLM 看到的是噪声</li>
              <li><strong>只有 CLIP + projection,没有 instruction tuning</strong>:模型只会描述图像,不会按指令回答</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
