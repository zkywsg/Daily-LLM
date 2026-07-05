import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEIT_SOURCE_PATH } from "../lib/prose";
import { RecipeCompareTable } from "../widgets/RecipeCompareTable";
import { DataEfficiencyChart } from "../widgets/DataEfficiencyChart";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function RecipeStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:强数据增强 + 现代训练 recipe
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ViT 直接在 ImageNet-1K 上训只有 76.5%,远低于 ResNet 的 78%。
        DeiT 的第一个答案很朴素:<strong>把现代 CNN(EfficientNet / ResNet-RS)训练 recipe
        系统性搬到 ViT 上</strong>——AdamW、cosine schedule、RandAugment、Mixup、CutMix、
        Stochastic depth、Repeated augmentation、EMA 一起上。ViT 没有 CNN 内置的
        translation equivariance,必须从增强里"显式"看到这些不变性。
      </p>

      <RecipeCompareTable />
      <p className={styles.caption}>
        ↑ 论文 Table 9 的系统消融:每一项贡献 0.5–2 分,加起来把 ViT-S 从 73%
        推到 79.8%(对比 ResNet-50 的 76.1%)。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <DataEfficiencyChart />
        <p className={styles.caption}>
          ↑ 原版 ViT 只在 JFT-300M(3 亿图像,Google 私有)预训练后才达到 77.9%;
          DeiT 只用 ImageNet-1K(130 万图像,公开)就训到 81.8% / 蒸馏后 83.4%。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DEIT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DEIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 ViT 比 CNN 更依赖增强
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.6, color: "var(--ink-primary)" }}>
              CNN 的卷积核自带 translation equivariance——同一个 pattern 挪个位置,
              卷积核照样能识别,这是"免费"的增强。ViT 的 attention 是全局的、无先验的,
              这种"挪位置也认得出"的能力必须靠数据增强(裁剪、翻转、Mixup、CutMix)显式教会。
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              这一发现后来被所有 ViT 系工作(Swin / BEiT / MAE / DINO)沿用。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
