import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VDM_SOURCE_PATH } from "../lib/prose";
import { JointTrainingWidget } from "../widgets/JointTrainingWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function JointTrainingStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:图像/视频联合训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        视频数据采集/标注成本远高于图像,联合训练让模型同时从大规模图像数据集和相对稀缺的视频数据集里学习。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={VDM_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <JointTrainingWidget />
        </div>
      </div>
    </div>
  );
}
