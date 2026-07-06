import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BLIP_SOURCE_PATH } from "../lib/prose";
import { QFormerArchitectureDiagram } from "../widgets/QFormerArchitectureDiagram";
import { TrainingCostCompareChart } from "../widgets/TrainingCostCompareChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function QFormerStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:BLIP-2 — Q-Former 冻结大模型 + 轻量桥接
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        BLIP-2 完全冻结视觉编码器和 LLM,只训练中间一个 Q-Former 模块。32 个
        可学习的 Query tokens 通过 cross-attention 查询视觉特征,提炼出"图像-
        语言对齐"的向量,再喂给冻结的 LLM 生成文本。整个训练只更新 12B 总参数
        里的 188M(1.5%)。
      </p>

      <QFormerArchitectureDiagram />
      <p className={styles.caption}>
        ↑ 视觉编码器和 LLM 都用❄标记冻结,只有中间 Q-Former 可训练。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <TrainingCostCompareChart />
        <p className={styles.caption}>
          ↑ Flamingo-80B 需要 1500 TPU × 10 天,BLIP-2 只需 16 A100 × 9 天 — 训练成本降约 100×。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={BLIP_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={BLIP_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,BLIP-2 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有三任务联合,没有 CapFilt</strong>:网络数据噪声严重,COCO CIDEr 从 133.3 跌到 117.5</li>
              <li><strong>只有 CapFilt,没有 Q-Former 冻结大模型</strong>:仍要从零训完整 VLM,只有大公司玩得起</li>
              <li><strong>只有 Q-Former,没有三任务对齐能力</strong>:Q tokens 学不到语言相关信息,LLM 拿到的是噪声</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
