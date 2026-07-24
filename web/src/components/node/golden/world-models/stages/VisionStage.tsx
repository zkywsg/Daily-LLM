import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORLD_MODELS_SOURCE_PATH } from "../lib/prose";
import { VaeCompressWidget } from "../widgets/VaeCompressWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function VisionStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:V(Vision)—— VAE 视觉压缩
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把高维原始帧压缩成低维潜向量 z,后续 M 和 C 全部在这个压缩后的潜空间里工作,而不是直接处理像素。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <VaeCompressWidget />
        </div>
      </div>
    </div>
  );
}
