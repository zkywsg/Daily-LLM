import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORLD_MODELS_SOURCE_PATH } from "../lib/prose";
import { MixtureDensityWidget } from "../widgets/MixtureDensityWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function MemoryStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:M(Memory)—— MDN-RNN 时序动态预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        M 不是预测确定的下一状态,而是预测一个混合高斯分布——世界是有随机性的,同一个当前状态可能走向多种不同的未来。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WORLD_MODELS_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <MixtureDensityWidget />
        </div>
      </div>
    </div>
  );
}
