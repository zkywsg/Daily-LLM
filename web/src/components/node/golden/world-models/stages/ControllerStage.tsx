import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORLD_MODELS_SOURCE_PATH } from "../lib/prose";
import { DreamRolloutWidget } from "../widgets/DreamRolloutWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ControllerStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:C(Controller)—— 完全在梦境里训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        C 是一个极小的线性控制器,用进化策略(而非梯度下降)训练,训练时用到的所有"经验"都来自 M 自回归生成的梦境轨迹,不是真实环境交互。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <DreamRolloutWidget />
        </div>
      </div>
    </div>
  );
}
