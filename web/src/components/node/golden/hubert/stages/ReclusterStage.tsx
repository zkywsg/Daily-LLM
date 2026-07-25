import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { HUBERT_SOURCE_PATH } from "../lib/prose";
import { IterativeReclusterWidget } from "../widgets/IterativeReclusterWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ReclusterStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:迭代式重新聚类
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用前一轮模型的隐藏层特征重新聚类,生成质量更高、更贴近音素边界的新伪标签,通常迭代 2 轮。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <IterativeReclusterWidget />
        </div>
      </div>
    </div>
  );
}
