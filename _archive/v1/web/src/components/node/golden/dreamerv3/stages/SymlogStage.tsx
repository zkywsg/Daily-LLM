import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DREAMERV3_SOURCE_PATH } from "../lib/prose";
import { SymlogWidget } from "../widgets/SymlogWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function SymlogStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:symlog 归一化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        不同任务领域的 reward/输出值天差地别,symlog 把它们统一压缩到可比范围,是固定同一套超参数跨领域通吃的关键工程细节。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DREAMERV3_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <SymlogWidget />
        </div>
      </div>
    </div>
  );
}
