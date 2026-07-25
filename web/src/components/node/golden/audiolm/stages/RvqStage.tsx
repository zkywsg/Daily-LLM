import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUDIOLM_SOURCE_PATH } from "../lib/prose";
import { RvqWidget } from "../widgets/RvqWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function RvqStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:声学 token —— 残差量化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用神经编解码器把音频压缩成多层残差量化 token,负责重建出高保真的波形细节。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={AUDIOLM_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <RvqWidget />
        </div>
      </div>
    </div>
  );
}
