import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { HUBERT_SOURCE_PATH } from "../lib/prose";
import { ClusteringWidget } from "../widgets/ClusteringWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ClusteringStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:离线聚类生成伪标签
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用独立的、不依赖被训练模型本身的聚类步骤生成一套固定的伪标签,避免 Wav2Vec 2.0 那种联合优化目标带来的训练不稳定。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <ClusteringWidget />
        </div>
      </div>
    </div>
  );
}
