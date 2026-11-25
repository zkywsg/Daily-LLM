import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GENIE_SOURCE_PATH } from "../lib/prose";
import { LamWidget } from "../widgets/LamWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function LamStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Latent Action Model(LAM)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        从海量无标注视频里,仅凭相邻帧的变化就能无监督推断出一套离散动作空间,不需要任何人工标注的动作数据。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GENIE_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <LamWidget />
        </div>
      </div>
    </div>
  );
}
