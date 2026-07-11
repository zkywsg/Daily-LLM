import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ADAPTER_SOURCE_PATH } from "../lib/prose";
import { BOTTLENECK_DIM } from "../lib/data";
import { AdapterInsertionDiagram } from "../widgets/AdapterInsertionDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function BottleneckStructureStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Bottleneck 结构 + 插入位置
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        全参微调每个任务要更新 BERT 全部 {`${(BOTTLENECK_DIM.d)}`} 维隐藏表示上的
        340M 参数。Houlsby 等人在每层插入一个 bottleneck 模块:down-project 把
        d={BOTTLENECK_DIM.d} 压到 r={BOTTLENECK_DIM.r},过一个非线性,再 up-project
        撑回 d 维 —— r ≪ d 把每层新增参数压到 ~98K,12 层仅 ~1.2M / 任务。
      </p>

      <AdapterInsertionDiagram />
      <p className={styles.caption}>
        ↑ Adapter 在每层插入两次:Attention 后一次、FFN 后一次;内部结构是
        down(d→r) → 非线性 → up(r→d),外加 residual。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={ADAPTER_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={ADAPTER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 r ≪ d 这么关键
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`d = ${BOTTLENECK_DIM.d}, r = ${BOTTLENECK_DIM.r}
每层参数 = 2 × d × r
        = 2 × ${BOTTLENECK_DIM.d} × ${BOTTLENECK_DIM.r}
        ≈ ${Math.round(BOTTLENECK_DIM.paramsPerLayer / 1000)}K
12 层 ≈ ${(BOTTLENECK_DIM.paramsPerTask / 1e6).toFixed(1)}M / 任务
BERT-large 340M 的 0.36%`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              如果不做 bottleneck(满秩 adapter),参数量会等同一层 FFN(约 36M),
              压缩直接失败。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
