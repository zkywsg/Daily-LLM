import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CONVNEXT_SOURCE_PATH } from "../lib/prose";
import { ModernizationWaterfallDiagram } from "../widgets/ModernizationWaterfallDiagram";
import { TrainingRecipeCompareTable } from "../widgets/TrainingRecipeCompareTable";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function TrainingRecipeStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:训练 recipe 现代化 — 单步收益最大的一步(+2.7)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ConvNeXt 现代化路线图里最反直觉的发现:光把 ResNet-50 的训练 recipe 换成
        ViT 时代的现代配置(AdamW + 强增强 + 300 epoch),精度就从 76.1% 涨到 78.8%,
        比所有结构改造单步收益加起来还多——ViT 的胜利大半在训练 recipe,不在算子。
      </p>

      <ModernizationWaterfallDiagram />
      <p className={styles.caption}>
        ↑ 点击"下一步"逐项展示 ResNet-50 → ConvNeXt-T 的累积精度提升 — 训练 recipe
        现代化(+2.7)是单步收益最大的一项,占总改造(76.1% → 82.0%)的近一半。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={CONVNEXT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={CONVNEXT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              训练 recipe:旧 vs 新
            </div>
            <TrainingRecipeCompareTable />
          </div>
        </div>
      </div>
    </div>
  );
}
