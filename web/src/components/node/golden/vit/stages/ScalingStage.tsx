import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VIT_SOURCE_PATH } from "../lib/prose";
import { ScalingCurveCompare } from "../widgets/ScalingCurveCompare";
import { InductiveBiasCompare } from "../widgets/InductiveBiasCompare";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ScalingStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:大数据 + 大模型 → 反超 CNN 的关键
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        在 ImageNet-1k 上,ResNet 完胜 ViT(80.4 vs 77.9)—— 因为 CNN 自带
        locality + translation invariance 的归纳偏置。但只要数据集够大
        (ImageNet-21k / JFT-300M),ViT 反超 CNN —— 因为 attention 没先验,
        给够数据自己学到的表示比 CNN 内置的更通用。这是 ViT 论文最重要的结论。
      </p>

      <ScalingCurveCompare />
      <p className={styles.caption}>
        ↑ 横轴 = 预训练数据量(log)。1.3M 时 ResNet 领先 2.5 pt;14M 时
        ViT 反超 0.5 pt;300M 时 ViT 领先 2.7 pt。预训练数据量直接决定
        架构选型。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={VIT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={VIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <InductiveBiasCompare />
          <p className={styles.caption}>
            CNN 用强 prior 换"小数据可用",代价是天花板低;ViT 砍掉所有 prior,
            代价是小数据用不动,收益是大数据下持续提升 —— "scaling law" 的视觉版。
          </p>
        </div>
      </div>
    </div>
  );
}
