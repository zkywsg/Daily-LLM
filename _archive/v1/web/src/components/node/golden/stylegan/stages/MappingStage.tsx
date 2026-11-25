import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { STYLEGAN_SOURCE_PATH } from "../lib/prose";
import { TraditionalVsStyleGAN } from "../widgets/TraditionalVsStyleGAN";
import { PPLComparison } from "../widgets/PPLComparison";
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

export function MappingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [side, setSide] = useState<"trad" | "style" | "both">("both");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Mapping Network — z → 解纠缠的 W 空间
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        z 服从 Gaussian 分布,真实人脸分布不是 Gaussian(年轻人多/老人少等)。
        直接把 z 喂 G 会让 latent 各维度耦合。加 8 层 MLP 让 z→w 吸收扭曲,
        w 空间自然贴合真实数据形状,每个维度对应相对独立的语义属性。
      </p>

      <TraditionalVsStyleGAN side={side} />
      <p className={styles.caption}>
        ↑ 左:传统 GAN z 直接喂 G,latent 纠缠;右:StyleGAN z 经 8 层 MLP → w,
        G 输入是学到的常量 4×4,所有变化通过 AdaIN 注入。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("trad")} style={btnStyle(side === "trad")}>聚焦 传统 GAN</button>
        <button type="button" onClick={() => setSide("style")} style={btnStyle(side === "style")}>聚焦 StyleGAN</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={STYLEGAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={STYLEGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <PPLComparison />
          <p className={styles.caption}>
            ↑ Perceptual Path Length 越低 → latent 越"线性"。W 空间 PPL 是 Z 的一半,
            证明 W 更适合 latent editing — 走一步能平滑改变年龄/性别/表情。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么加 8 层 MLP 能解纠缠?
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>z 是 Gaussian,真实分布不是</li>
              <li>无 MLP:G 必须学扭曲映射 → 维度耦合</li>
              <li>有 MLP:扭曲被 MLP 吸收 → w 可自由贴合真实</li>
              <li>W 空间成为后续所有 latent editing 工作的基础</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
