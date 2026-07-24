import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAMENGEN_SOURCE_PATH } from "../lib/prose";
import { RlTrajectoryWidget } from "../widgets/RlTrajectoryWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function RlDataStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:RL agent 自动生成训练数据
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用强化学习 agent 自我博弈产出海量(帧,动作)训练对,替代成本高昂的人类录屏采集。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <RlTrajectoryWidget />
        </div>
      </div>
    </div>
  );
}
