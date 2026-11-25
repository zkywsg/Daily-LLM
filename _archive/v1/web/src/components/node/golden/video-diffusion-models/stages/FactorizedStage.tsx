import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VDM_SOURCE_PATH } from "../lib/prose";
import { FlopsCompareWidget } from "../widgets/FlopsCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function FactorizedStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:时空分解架构
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用 2D 空间卷积(逐帧)+ 1D 时间卷积(沿时间轴)替代昂贵的完整 3D 卷积,大幅降低视频生成的算力开销。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={VDM_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={VDM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <FlopsCompareWidget />
        </div>
      </div>
    </div>
  );
}
