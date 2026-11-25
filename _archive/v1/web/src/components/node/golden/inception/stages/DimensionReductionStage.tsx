import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { INCEPTION_SOURCE_PATH } from "../lib/prose";
import { BottleneckCompareChart } from "../widgets/BottleneckCompareChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DimensionReductionStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:1×1 卷积瓶颈 — "先压再算"防止通道爆炸
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        简单多分支并行有一个致命问题:通道数会随分支累积。Inception 的核心 trick 是
        在每个 3×3 / 5×5 之前先用 1×1 卷积做"瓶颈降维",把输入通道砍下来再算贵的卷积,
        参数量直接降一个数量级。
      </p>

      <BottleneckCompareChart />
      <p className={styles.caption}>
        ↑ 输入 256 通道、每分支输出 128 通道的场景下,5×5 分支直接算 vs 先用 1×1 压到 64 再算的参数量对比。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={INCEPTION_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              工程陷阱:顺序不能反
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
              <li>1×1 必须在 3×3 / 5×5 之前 —— 顺序反了 3×3 已经在高维输入上算完,1×1 只能压缩输出通道</li>
              <li>pool 分支的 1×1 放在 MaxPool 之后 —— MaxPool 不改通道,只需在 concat 前压一下</li>
              <li>1×1 只对通道维做线性组合(不改变空间维),计算便宜但能学到通道压缩</li>
            </ul>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-3) 0 0", lineHeight: 1.6 }}>
              这一"先压再算"的 bottleneck 结构后来在 ResNet bottleneck、MobileNet inverted residual、
              Transformer FFN(d → 4d → d)里被反复借用。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
