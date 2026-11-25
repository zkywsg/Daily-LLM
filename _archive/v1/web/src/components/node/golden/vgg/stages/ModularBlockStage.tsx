import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VGG_SOURCE_PATH } from "../lib/prose";
import { VggBlockDiagram } from "../widgets/VggBlockDiagram";
import { VggDepthProgressionChart } from "../widgets/VggDepthProgressionChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ModularBlockStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:模块化 block — 同 block 内通道不变,跨 block 翻倍
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        VGG 把网络组织成 5 个 conv block + 3 个 fc:每 block 内通道数固定,跨 block 用
        MaxPool/2 把空间减半、同时下一 block 通道翻倍。这种"空间换通道"的固定模式后来被
        ResNet / Inception / ViT 全部继承,成为 CNN 通用设计模板。
      </p>

      <VggBlockDiagram />
      <p className={styles.caption}>
        ↑ VGG-16 的 5 个 conv block:通道 64→128→256→512→512 翻倍再封顶,空间
        224→112→56→28→14→7 逐 block 减半。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={VGG_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              深度作为单一变量
            </div>
            <VggDepthProgressionChart />
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-2) 0 0", lineHeight: 1.6 }}>
              VGG-11 → VGG-19,唯一变化的是每个 block 内 conv 的重复次数——整网超参数
              只有"block 数"和"每 block 内 conv 重复次数"这 2 个,规整性是 VGG 的工程红利。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
