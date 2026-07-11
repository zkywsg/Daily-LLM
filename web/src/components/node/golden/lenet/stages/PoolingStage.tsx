import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LENET_SOURCE_PATH } from "../lib/prose";
import { ReceptiveFieldGrowthDiagram } from "../widgets/ReceptiveFieldGrowthDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function PoolingStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:平均池化降采样 — 感受野逐层放大
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        S2 把 6@28×28 降到 6@14×14,S4 把 16@10×10 降到 16@5×5。每经过一次池化,
        空间分辨率减半、计算量降 4 倍,更关键的是深层 neuron 在原图上能"看到"的范围翻倍——
        这是把"平移不变性"先验编码进网络的关键机制。
      </p>

      <ReceptiveFieldGrowthDiagram />
      <p className={styles.caption}>
        ↑ C1 → S2 → C3 → S4 → C5,每一次卷积 / 池化都让感受野在原图上覆盖更大范围。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={LENET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么是平均池化
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              LeNet 用的是平均池化,不是后来流行的 max pool。降采样层把"小幅度像素位移
              对输出影响很小"这条平移不变性先验直接编码进网络,和卷积层的局部+共享先验
              互补,缺一不可。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
