import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { EFFICIENTNET_SOURCE_PATH } from "../lib/prose";
import { MBConvBlockDiagram } from "../widgets/MBConvBlockDiagram";
import { B0StageTable } from "../widgets/B0StageTable";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function SeedArchitectureStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:NAS 搜出强种子 B0 — MBConv + SE + Swish
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        种子模型选对了,compound scaling 才有意义。EfficientNet-B0 是在"FLOPs ≈ 400M、
        精度最优"目标下用 NAS 搜出来的(和 MnasNet 同一套搜索空间),结构骨架是
        stem + 7 个 MBConv stage + head,如果 B0 本身不够强,后续 B1-B7 全部按公式放大也救不回精度。
      </p>

      <MBConvBlockDiagram />
      <p className={styles.caption}>
        ↑ MBConv 内部:1×1 升维(expand 6×)→ depthwise k×k → SE 门控 → 1×1 降维
        (线性瓶颈,无激活)+ shortcut。激活函数全网用 Swish/SiLU 替代 ReLU。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={EFFICIENTNET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              EfficientNet-B0 的 7 个 NAS stage
            </div>
            <B0StageTable />
          </div>
        </div>
      </div>
    </div>
  );
}
