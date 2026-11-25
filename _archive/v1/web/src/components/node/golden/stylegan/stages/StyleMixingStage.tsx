import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { STYLEGAN_SOURCE_PATH } from "../lib/prose";
import { LayerGranularity } from "../widgets/LayerGranularity";
import { StyleMixingDemo } from "../widgets/StyleMixingDemo";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

const PRESETS = [
  { label: "全 A",              split: 9 },
  { label: "A 主 + B 极细 (7)", split: 7 },
  { label: "A 粗中 + B 细 (4)", split: 4 },
  { label: "A 粗 + B 中细 (2)", split: 2 },
  { label: "全 B",              split: 0 },
];

export function StyleMixingStage({ mechanism3Prose, synergyProse }: Props) {
  const [split, setSplit] = useState(4);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:分层 Style + Style Mixing — 自然得到粒度分层
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        G 不接收 z 作为输入,从学到的常量 4×4 起步,所有变化通过 style 在每层注入。
        不同分辨率层控制不同语义粒度:粗(4-8²)姿态脸型 · 中(16-32²)发型眼神 ·
        细(64²+)肤色纹理。Style mixing 训练时随机拼 w_A + w_B,推理时可组合两人的不同粒度属性。
      </p>

      <LayerGranularity splitLayer={split} />
      <p className={styles.caption}>
        ↑ 9 层 style 注入,颜色区分粒度。切换点(粉→绿分界)表示 style mixing
        在哪一层切换 w_A → w_B。滑动改变切换位置看效果。
      </p>

      <StyleMixingDemo splitLayer={split} />
      <p className={styles.caption}>
        ↑ 用抽象人脸演示 mixing 结果。split=2 时 A 只贡献脸型 · split=4 时 A 还贡献发型 ·
        split=7 时 A 贡献主体 B 仅贡献光线。真实 StyleGAN 上就是"A 姿态 + B 肤色"这类可控组合。
      </p>

      <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Style mixing 切换点(前 N 层用 A · 后续用 B)
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {PRESETS.map((p) => (
            <button key={p.label} type="button" onClick={() => setSplit(p.split)} style={btnStyle(split === p.split)}>{p.label}</button>
          ))}
        </div>
        <input type="range" min={0} max={9} step={1} value={split} onChange={(e) => setSplit(parseInt(e.target.value))} style={{ width: "100%", marginTop: 12 }} />
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 6, textAlign: "center" }}>
          split = {split} · 前 {split} 层用 w_A · 后 {9 - split} 层用 w_B
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={STYLEGAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={STYLEGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              Per-pixel Noise Input
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              每层额外加 <code>x' = x + learned_scale · noise</code>。
              直觉:头发走向 / 毛孔分布 / 痘痘位置等 stochastic detail 应该独立于 style,
              从 noise 直接产生而非压缩到 z 里。这让 G 学得更精细,同一 w 下换 noise
              产生"同一个人不同瞬间"的合理变化。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
