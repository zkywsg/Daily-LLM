import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CONVNEXT_SOURCE_PATH } from "../lib/prose";
import { NormActCountDiagram } from "../widgets/NormActCountDiagram";
import { CONVNEXT_VS_SWIN } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function NormActStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Norm/Act 极简化 — 每个 block 只留 1 个 LN + 1 个 GELU
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ResNet 时代一个 Bottleneck block 堆 3 组 BN + 3 个 ReLU;Transformer block
        只有 1 次 LN(attention 前)+ 1 次 LN(FFN 前)+ 1 次 GELU(FFN 中间)。
        ConvNeXt 照搬这个极简思路,少堆几层 norm/act,精度反而涨 0.5 个点——
        到处堆 BN + ReLU 的做法是有冗余的。
      </p>

      <NormActCountDiagram />
      <p className={styles.caption}>
        ↑ ResNet Bottleneck 每 block 3 组 BN+ReLU,ConvNeXt/Transformer 每 block
        只留 1 个 LN + 1 个 GELU——LN 对 batch size 不敏感、无需切换 running mean、
        分布式训练无同步开销,工程价值远大于精度本身的 0.1 个点。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={CONVNEXT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={CONVNEXT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              ConvNeXt-T vs Swin-T(同算力)
            </div>
            {CONVNEXT_VS_SWIN.map((row) => (
              <div key={row.model} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: row.highlight ? "#ec4899" : "var(--ink-primary)" }}>
                  {row.model}
                </div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", lineHeight: 1.5 }}>
                  参数 {row.params} · FLOPs {row.flops} · Top-1 {row.top1}%
                  {row.throughput ? ` · ${row.throughput} img/s` : ""}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
