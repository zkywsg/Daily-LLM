import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LDM_SOURCE_PATH } from "../lib/prose";
import { PixelVsLatentCost } from "../widgets/PixelVsLatentCost";
import { VaeReconstructionGrid } from "../widgets/VaeReconstructionGrid";
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

export function PerceptualCompressionStage({ intuitionProse, mechanism1Prose }: Props) {
  const [resolution, setResolution] = useState<256 | 512 | 1024>(512);
  const [vaeKind, setVaeKind] = useState<"l2" | "perceptual">("perceptual");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Perceptual Compression — VAE 把 512² 压到 64²
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DDPM 在 512² pixel 上 1000 步采样 = 786M 次卷积,V100 几十秒一张。
        LDM Stage 1 独立训一个 perceptual + adversarial VAE,把图压到 64²×4 latent
        (49× 压缩),Stage 2 完全冻结 VAE,只在 latent 上跑 diffusion。
      </p>

      <PixelVsLatentCost resolution={resolution} />
      <p className={styles.caption}>
        ↑ Pixel 粉条占满整宽度,Latent 绿条几乎不可见 — 1024² 时差距 49×。
        diffusion 数学完全不变,只是搬到了一个 49× 更小的空间。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={LDM_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={LDM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <VaeReconstructionGrid vaeKind={vaeKind} />
          <p className={styles.caption}>
            ↑ 左原图 → 中 latent grid → 右 decode 结果。纯 L2 训出来糊,
            perceptual + GAN 训出来保细节 — 这是 LDM Stage 1 能成立的关键。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              VAE 训练损失
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button type="button" onClick={() => setVaeKind("l2")} style={btnStyle(vaeKind === "l2")}>L2 baseline</button>
              <button type="button" onClick={() => setVaeKind("perceptual")} style={btnStyle(vaeKind === "perceptual")}>LPIPS + GAN</button>
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 8, lineHeight: 1.4 }}>
              SD 的 VAE 用 L1 + LPIPS + PatchGAN-adversarial,~84M 参数,在 LAION-2B 上几周训出来一次,所有后续 SD 版本 / fine-tune 都复用。
            </div>

            <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px dashed var(--border)" }}>
              <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                输入分辨率
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                {([256, 512, 1024] as const).map((r) => (
                  <button key={r} type="button" onClick={() => setResolution(r)} style={btnStyle(resolution === r)}>{r}²</button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
