import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DCGAN_SOURCE_PATH } from "../lib/prose";
import { ArchitectureDiagram } from "../widgets/ArchitectureDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function AllConvStage({ intuitionProse, mechanism1Prose }: Props) {
  const [mode, setMode] = useState<"mlp" | "dcgan">("dcgan");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:全卷积架构 — strided conv / transposed conv 替代 pool 和 fc
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        原版 GAN 是 MLP,fc 层把 z reshape 成图像、把图像 flatten 成 scalar,
        破坏了空间结构,加深加宽就崩溃。DCGAN 把 D 的下采样换成 strided conv、
        G 的上采样换成 transposed conv,不用 max-pool,让网络自己学采样方式,
        整个网络全程在空间网格上操作。
      </p>

      <ArchitectureDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看原版 GAN 的 MLP 结构 vs DCGAN 的全卷积结构 —— DCGAN 的每一层都保留空间网格,
        原版 GAN 的每一层都是打平的向量。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("dcgan")} style={btnStyle(mode === "dcgan")}>DCGAN(全卷积)</button>
        <button type="button" onClick={() => setMode("mlp")} style={btnStyle(mode === "mlp")}>原版 GAN(MLP)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DCGAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DCGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么去掉 fc 层
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              Pooling 是固定不可学的算子,fc 层在输入输出端把张量打平,两者都破坏图像的空间结构。
              DCGAN 直接从 1×1×100 噪声 conv 出图像、把图像 conv 到 1×1×1 输出,
              让网络自己学下 / 上采样方式,和图像本身的空间结构对齐。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
