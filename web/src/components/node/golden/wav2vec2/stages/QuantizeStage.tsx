import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WAV2VEC2_SOURCE_PATH } from "../lib/prose";
import { ContrastiveWidget } from "../widgets/ContrastiveWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function QuantizeStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:量化模块 + 对比学习目标
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        连续特征被量化成离散码本向量作为对比学习的目标,模型需要从候选码本里正确识别出真实目标。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WAV2VEC2_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <ContrastiveWidget />
        </div>
      </div>
    </div>
  );
}
