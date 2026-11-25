import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GENIE_SOURCE_PATH } from "../lib/prose";
import { TokenizerWidget } from "../widgets/TokenizerWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function TokenizerStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:视频 tokenizer
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把原始帧压缩成离散 token 序列,是后续无监督推断动作、自回归生成下一帧的共同基础表示。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GENIE_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GENIE_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <TokenizerWidget />
        </div>
      </div>
    </div>
  );
}
