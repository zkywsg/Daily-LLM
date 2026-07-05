import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEIT_SOURCE_PATH } from "../lib/prose";
import { BenchmarkBars } from "../widgets/BenchmarkBars";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function LongTrainingStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:300 Epoch 长训练 + 单卡可复现
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DeiT 训练 300 epoch——典型 ViT 训练的 3 倍。ViT 在小数据上 underfits,
        需要更长时间才能吃透 1.3M 图像;强增强让每个 epoch 看到的图像因增强不同
        而"变新",300 epoch 相当于看了 3.9 亿张"虚拟图像"。关键是这套训练
        单张 V100 训 4 天就能跑完,学界研究者不需要 TPU 集群也能复现。
      </p>

      <BenchmarkBars />
      <p className={styles.caption}>
        ↑ 论文 Table 1:DeiT-B with distillation(83.4%)用 1 张 V100 训 4 天,
        击败训练成本高 100 倍的 EfficientNet-B7(82.9%,32 TPU × 3 周)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DEIT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DEIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              三件套缺一不可
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.6, color: "var(--ink-primary)" }}>
              只有 recipe 没有 distill → 81.8%,输给 EfficientNet-B7。
              只有 distill 没有 recipe → underfit 救不回,~78%。
              只有 recipe + distill 没有长训练 → 严重 underfit,~80%。
              三者合起来才是 83.4%。
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              和 ResNet 的 shortcut + BN + He 初始化协同关系一致——单一改进不成立,组合才成立。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
