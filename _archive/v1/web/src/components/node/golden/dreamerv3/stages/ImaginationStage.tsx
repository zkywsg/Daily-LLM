import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DREAMERV3_SOURCE_PATH } from "../lib/prose";
import { ImaginationRolloutWidget } from "../widgets/ImaginationRolloutWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ImaginationStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:纯 latent imagination 训练 actor-critic
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        actor 和 critic 完全在世界模型生成的想象轨迹上训练,不需要额外调用真实环境采样,大幅提升样本效率。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <ImaginationRolloutWidget />
        </div>
      </div>
    </div>
  );
}
