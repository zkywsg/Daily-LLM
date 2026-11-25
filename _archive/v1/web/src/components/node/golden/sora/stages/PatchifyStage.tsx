import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SORA_SOURCE_PATH } from "../lib/prose";
import { SpacetimePatchWidget } from "../widgets/SpacetimePatchWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function PatchifyStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:视频压缩网络 + spacetime patches
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把视频压缩到低维时空潜空间后切成 patch 序列,不同长宽比/时长的输入统一表示成变长 token 序列。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SORA_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SORA_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SpacetimePatchWidget />
        </div>
      </div>
    </div>
  );
}
