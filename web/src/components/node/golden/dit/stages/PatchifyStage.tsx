import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DIT_SOURCE_PATH } from "../lib/prose";
import { PatchifyPipeline } from "../widgets/PatchifyPipeline";
import { UnetVsDitArchitecture } from "../widgets/UnetVsDitArchitecture";
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

export function PatchifyStage({ intuitionProse, mechanism1Prose }: Props) {
  const [patchSize, setPatchSize] = useState<2 | 4 | 8>(2);
  const [side, setSide] = useState<"unet" | "dit" | "both">("both");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Patchify Latents — 把 VAE latent 当 token 序列
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DiT 站在 LDM 肩膀上,所有操作都在 VAE latent (32×32×4) 而非 pixel。
        第一步 patchify 用 patch size p 切块,变成 (32/p)² 个 token,
        线性投影到 d 维 + 2D 位置编码。p 越小 token 越多,FLOPs 平方增长,质量更好。
      </p>

      <PatchifyPipeline patchSize={patchSize} />
      <p className={styles.caption}>
        ↑ 切换 patch_size 看 token 数变化。p=2 给 256 token (DiT-XL 默认),
        p=4 给 64,p=8 给 16。FLOPs ∝ token² 所以 p 是关键工程旋钮。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        {([2, 4, 8] as const).map((p) => (
          <button key={p} type="button" onClick={() => setPatchSize(p)} style={btnStyle(patchSize === p)}>p={p}</button>
        ))}
      </div>

      <UnetVsDitArchitecture side={side} />
      <p className={styles.caption}>
        ↑ U-Net 的 encoder-decoder + skip 漏斗,vs DiT 的 patchify → Transformer × N → unpatchify
        线性流。整套 DiT 实现 ~250 行,比 U-Net 简洁很多。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("unet")} style={btnStyle(side === "unet")}>聚焦 U-Net</button>
        <button type="button" onClick={() => setSide("dit")} style={btnStyle(side === "dit")}>聚焦 DiT</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DIT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              patch_size 对 token 数 / FLOPs 影响
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>p</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>token 数</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>FLOPs (相对)</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>说明</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>2</td>
                  <td style={{ padding: "8px" }}>256</td>
                  <td style={{ padding: "8px" }}>1.0 (默认)</td>
                  <td style={{ padding: "8px" }}>最佳质量</td>
                </tr>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>4</td>
                  <td style={{ padding: "8px" }}>64</td>
                  <td style={{ padding: "8px" }}>~0.06</td>
                  <td style={{ padding: "8px" }}>更快但 FID 涨</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px" }}>8</td>
                  <td style={{ padding: "8px" }}>16</td>
                  <td style={{ padding: "8px" }}>~0.004</td>
                  <td style={{ padding: "8px" }}>太粗 · 细节丢</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              512² 输入时 latent 64×64,p=2 给 1024 token,FLOPs 涨 16× —
              这就是 Sora 之类高分辨率 DiT 工程上需要稀疏注意力的根因。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
