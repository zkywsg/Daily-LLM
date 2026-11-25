import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RESNET_SOURCE_PATH } from "../lib/prose";
import { DepthSlider } from "../widgets/DepthSlider";
import { DegradationCurves } from "../widgets/DegradationCurves";
import { ImageNetMilestones } from "../widgets/ImageNetMilestones";
import styles from "./Stage.module.css";

interface Props {
  previousWorkProse: string;
  coreInsightProse: string;
  depth: number;
  onDepthChange: (d: number) => void;
}

export function DegradationStage({
  previousWorkProse,
  coreInsightProse,
  depth,
  onDepthChange,
}: Props) {
  return (
    <div className={styles.grid}>
      <div>
        <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
          前作进展
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer
            markdown={previousWorkProse}
            sourcePath={RESNET_SOURCE_PATH}
          />
        </div>

        <h2
          style={{
            fontSize: "var(--fs-2xl)",
            marginTop: "var(--space-8)",
            marginBottom: "var(--space-4)",
          }}
        >
          核心思想
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer
            markdown={coreInsightProse}
            sourcePath={RESNET_SOURCE_PATH}
          />
        </div>

        <DepthSlider value={depth} onChange={onDepthChange} />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动选择网络深度，右图实时变化。深度超过 30 层后，plain 网络会出现"先降后升"的退化形态。
        </p>
      </div>

      <div className={styles.stickyPanel}>
        <DegradationCurves depth={depth} />
        <div style={{ marginTop: "var(--space-6)" }}>
          <ImageNetMilestones />
        </div>
      </div>
    </div>
  );
}
