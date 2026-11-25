import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DENSENET_SOURCE_PATH } from "../lib/prose";
import { ChannelGrowthChart } from "../widgets/ChannelGrowthChart";
import { DENSENET121_BLOCKS, GROWTH_RATE_K } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function GrowthRateStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
        Growth Rate k 控制通道线性增长
      </h2>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DENSENET_SOURCE_PATH} />
          </div>

          <div style={{ marginTop: "var(--space-6)", display: "flex", flexWrap: "wrap", gap: "var(--space-2)" }}>
            {DENSENET121_BLOCKS.map((b, i) => (
              <div
                key={`${b.name}-${i}`}
                style={{
                  padding: "var(--space-2) var(--space-3)",
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-subtle)",
                  fontSize: "var(--fs-sm)",
                }}
              >
                {b.name}:{b.layers} 层
              </div>
            ))}
          </div>

          <p className={styles.caption}>
            DenseNet-121 由 4 个 Dense block(6/12/24/16 层)串联,growth rate 统一取 k={GROWTH_RATE_K}。
            block 之间用 transition layer 下采样并压缩通道数。
          </p>
        </div>

        <div className={styles.stickyPanel}>
          <ChannelGrowthChart />
          <p className={styles.caption}>
            悬停任意点查看该层的累积输入通道数。线性增长(而非指数)是 growth rate 机制的核心效果。
          </p>
        </div>
      </div>
    </div>
  );
}
