import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VDM_SOURCE_PATH } from "../lib/prose";
import { SlidingWindowWidget } from "../widgets/SlidingWindowWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ExtensionStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:条件生成的引导技术
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        reconstruction guidance 配合自回归滑窗,让模型能生成超出单次训练窗口长度的更长视频。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={VDM_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={VDM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SlidingWindowWidget />
        </div>
      </div>
    </div>
  );
}
