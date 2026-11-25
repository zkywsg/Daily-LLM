import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CONVNEXT_SOURCE_PATH } from "../lib/prose";
import { ConvNeXtBlockDiagram } from "../widgets/ConvNeXtBlockDiagram";
import { CONVNEXT_BLOCK_STEPS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function StructuralModernizationStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:结构现代化 — depthwise 7×7 + inverted bottleneck + patchify
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        按 Swin 的每个设计逐项移植:patchify stem 让不重叠 patch 成为可能、
        depthwise 7×7 控制 FLOPs 同时逼近 Swin 的 window size、inverted bottleneck
        借自 MobileNet v2 / Transformer FFN。整个 ConvNeXt Block 和 Transformer 的
        FFN block 一一对应——DWConv 7×7 是"局部 token mixer",两次 PWConv + GELU
        就是标准 MLP。
      </p>

      <ConvNeXtBlockDiagram />
      <p className={styles.caption}>
        ↑ ConvNeXt Block 内部:DWConv 7×7 → LN → PWConv↑(4× hidden)→ GELU → PWConv↓ →
        + shortcut。右侧标注每一步对应 Transformer block 里的哪个组件。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={CONVNEXT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              Block 内 6 步 ↔ Transformer 角色
            </div>
            {CONVNEXT_BLOCK_STEPS.map((step) => (
              <div key={step.label} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "var(--ink-primary)" }}>{step.label}</div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", lineHeight: 1.5 }}>{step.detail}</div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-secondary)", lineHeight: 1.5 }}>{step.role}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
