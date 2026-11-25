import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DREAMERV3_SOURCE_PATH } from "../lib/prose";
import { CategoricalLatentWidget } from "../widgets/CategoricalLatentWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function RssmStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:RSSM 世界模型 —— 离散隐变量
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用多组离散类别分布(而非连续高斯)表示隐状态,组合数随组数指数增长,能表达更丰富的不确定性和多模态未来。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <CategoricalLatentWidget />
        </div>
      </div>
    </div>
  );
}
