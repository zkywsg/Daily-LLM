import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SORA_SOURCE_PATH } from "../lib/prose";
import { ScaleQualityWidget } from "../widgets/ScaleQualityWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ScalingStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Diffusion Transformer 主干规模化到视频
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        延续 DiT 的规模化规律,把同一套 Transformer 主干直接放大到视频这个更高维的数据模态上。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SORA_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <ScaleQualityWidget />
        </div>
      </div>
    </div>
  );
}
