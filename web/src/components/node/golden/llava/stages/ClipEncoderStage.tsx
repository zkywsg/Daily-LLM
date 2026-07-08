import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LLAVA_SOURCE_PATH } from "../lib/prose";
import { ArchitectureDiagram } from "../widgets/ArchitectureDiagram";
import { TrainingCostCompareChart } from "../widgets/TrainingCostCompareChart";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ClipEncoderStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Frozen CLIP Vision Encoder — 复用已有视觉表示
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Flamingo / GPT-4V 那种深度多模态融合需要从零联合预训练,算力门槛
        极高。LLaVA 反问:能不能用 CLIP 的 ViT 当眼睛、LLaMA 当脑子,只训
        一个 projection 桥?LLaVA 用 CLIP ViT-L/14 把图像编码成 256 个
        patch token,整个 vision encoder 完全冻结 — CLIP 已经把图像和
        语言对齐过,不需要重新训练。
      </p>

      <ArchitectureDiagram />
      <p className={styles.caption}>
        ↑ 三个组件的可训练性差异一目了然 — 只有中间的 projection 是唯一从零训练的部分。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <TrainingCostCompareChart />
        <p className={styles.caption}>
          ↑ 冻结 CLIP + 极简 projection 让训练成本从 Flamingo 的 $1M+ 压到 LLaVA 的 $200。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={LLAVA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={LLAVA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              冻结 CLIP 的好处
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>省 90% 训练算力</li>
              <li>保留 CLIP 已学到的"语义概念"先验</li>
              <li>与所有 CLIP-based 应用(SD / search / classification)共享同一套视觉表示</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
