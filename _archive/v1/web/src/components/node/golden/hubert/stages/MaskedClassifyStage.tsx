import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { HUBERT_SOURCE_PATH } from "../lib/prose";
import { ClassDistributionWidget } from "../widgets/ClassDistributionWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function MaskedClassifyStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:BERT 式掩码预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        在被 mask 的位置用分类头预测伪标签类别,用标准交叉熵训练——不需要对比学习里的负采样。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={HUBERT_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <ClassDistributionWidget />
        </div>
      </div>
    </div>
  );
}
