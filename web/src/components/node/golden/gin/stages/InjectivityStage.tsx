import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GIN_SOURCE_PATH } from "../lib/prose";
import { InjectivityCounterexampleWidget } from "../widgets/InjectivityCounterexampleWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function InjectivityStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:为什么 mean / max 聚合不是单射的
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        单射(injective)意味着不同的输入永远映射到不同的输出。mean/max 会把"两个邻居都是 1"和"四个邻居都是 1"这两种明显不同的情况聚合成完全相同的结果 —— 丢失了"有多少个邻居"这个信息。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GIN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <InjectivityCounterexampleWidget />
          <p className={styles.caption}>↑ sum 能区分两个多重集,mean/max 不能 —— 这是 GIN 选 sum 的原因</p>
        </div>
      </div>
    </div>
  );
}
