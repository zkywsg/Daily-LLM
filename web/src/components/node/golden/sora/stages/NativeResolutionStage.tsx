import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SORA_SOURCE_PATH } from "../lib/prose";
import { NativeResolutionWidget } from "../widgets/NativeResolutionWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function NativeResolutionStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:原生分辨率/长宽比/时长训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        不强制裁剪/缩放到统一尺寸,直接在原生分辨率/长宽比/时长上训练,变长 patch 序列天然支持这种灵活性。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SORA_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SORA_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <NativeResolutionWidget />
        </div>
      </div>
    </div>
  );
}
