import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MUSICGEN_SOURCE_PATH } from "../lib/prose";
import { EncodecRvqWidget } from "../widgets/EncodecRvqWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function EncodecStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:EnCodec 残差量化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用神经音频编解码器把音频压缩成多层残差量化码本,K 层组合起来能重建出接近原始质量的音频。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <EncodecRvqWidget />
        </div>
      </div>
    </div>
  );
}
