import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MUSICGEN_SOURCE_PATH } from "../lib/prose";
import { DelayPatternWidget } from "../widgets/DelayPatternWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DelayPatternStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:码本交错(codebook interleaving)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把多个并行码本流按延迟错位规则重新排列成一条单一序列,单阶段模型就能同时建模所有层。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={MUSICGEN_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <DelayPatternWidget />
        </div>
      </div>
    </div>
  );
}
