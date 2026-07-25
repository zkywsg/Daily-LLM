import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WHISPER_SOURCE_PATH } from "../lib/prose";
import { DataFilterWidget } from "../widgets/DataFilterWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DataFilterStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:大规模弱监督数据收集与过滤
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        68 万小时数据质量参差不齐,用启发式规则和分类器过滤掉可能是机器生成的低质量转写,尽量保留高质量监督信号。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WHISPER_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <DataFilterWidget />
        </div>
      </div>
    </div>
  );
}
