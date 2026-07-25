import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WHISPER_SOURCE_PATH } from "../lib/prose";
import { SpectrogramWidget } from "../widgets/SpectrogramWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SpectrogramStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:标准 Transformer encoder-decoder + log-mel 频谱输入
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        整体架构没有任何语音专用的特殊设计,刻意选用标准架构以验证"数据规模而非架构精巧"才是关键因素。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SpectrogramWidget />
        </div>
      </div>
    </div>
  );
}
