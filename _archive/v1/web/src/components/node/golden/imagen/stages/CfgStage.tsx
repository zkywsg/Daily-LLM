import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { IMAGEN_SOURCE_PATH } from "../lib/prose";
import { CfgDataflow } from "../widgets/CfgDataflow";
import { CfgScaleCurves } from "../widgets/CfgScaleCurves";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

type Hl = "cond" | "uncond" | "diff" | "scale" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function CfgStage({ intuitionProse, mechanism1Prose }: Props) {
  const [hl, setHl] = useState<Hl>(null);
  const [w, setW] = useState(7.5);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Classifier-Free Guidance — 用减法 + 放大控制条件强度
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Diffusion 条件采样默认给出的是条件分布上的"平均预测",会混合"严格按 prompt"和"模糊接近 prompt"。
        CFG 训练时让模型 10-20% 概率丢条件,同时学条件预测和无条件预测。
        采样时取 [ε(条件) − ε(无条件)] 作为"条件特有方向",放大 w 倍加回去,
        让生成的图更严格符合 prompt。
      </p>

      <CfgDataflow highlight={hl} />
      <p className={styles.caption}>
        ↑ 每步采样要算两次 U-Net forward(条件 + 无条件)。点按钮聚焦各阶段。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("cond")} style={btnStyle(hl === "cond")}>条件预测</button>
        <button type="button" onClick={() => setHl("uncond")} style={btnStyle(hl === "uncond")}>无条件预测</button>
        <button type="button" onClick={() => setHl("diff")} style={btnStyle(hl === "diff")}>条件特有方向</button>
        <button type="button" onClick={() => setHl("scale")} style={btnStyle(hl === "scale")}>放大 w</button>
      </div>

      <CfgScaleCurves w={w} />
      <p className={styles.caption}>
        ↑ 拖动 w(SD WebUI 里的 CFG Scale 滑杆)看 fidelity/creativity/distortion 三者权衡。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>cfg_scale (w)</span><strong>{w.toFixed(1)}</strong>
      </label>
      <input type="range" min={0} max={20} step={0.5} value={w}
             onChange={(e) => setW(parseFloat(e.target.value))} style={{ width: "100%" }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={IMAGEN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={IMAGEN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              CFG 已成为所有现代 diffusion 标配
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>SD / SDXL / SD3 / Flux 全部默认开启</li>
              <li>DALL-E 2/3、Midjourney、Ideogram 同样依赖</li>
              <li>思想外溢到文本/3D/视频/音频生成</li>
              <li>代价:采样速度变 2×(每步 2 次 forward)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
