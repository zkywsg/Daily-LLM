import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { IMAGEN_SOURCE_PATH } from "../lib/prose";
import { EncoderScalingChart } from "../widgets/EncoderScalingChart";
import { ENCODER_SCALING } from "../lib/data";
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

export function EncoderStage({ mechanism3Prose, synergyProse }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:大文本编码器(T5-XXL)— 文本理解比视觉建模更重要
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Imagen 论文最重要的实证发现:固定 diffusion U-Net 大小,只变化 text encoder,
        FID 从 CLIP 的 12.1 一路降到 T5-XXL 的 7.27。这颠覆了之前的常识 —
        文生图瓶颈不在视觉建模,而在文本理解。CLIP 是对比学习产物,擅长图文匹配相似度,
        但没学到否定/比较/数量/空间关系等语言细节;T5 是纯语言模型,真正理解 prompt。
      </p>

      <EncoderScalingChart highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 6 个文本编码器规模对比,T5-XXL 比 CLIP 大 28×,FID 改善 40%。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {ENCODER_SCALING.map((e, i) => (
          <button key={i} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{e.name}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={IMAGEN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={IMAGEN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              后续继承者
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>SDXL</strong>(2023):双文本编码器(OpenCLIP + CLIP-G)</li>
              <li><strong>SD3 / Flux</strong>(2024):T5-XXL + CLIP,几乎照搬 Imagen 选择</li>
              <li><strong>DeepFloyd IF</strong>(2023):Imagen 开源复现,也用 T5-XXL</li>
              <li>Sora 用 T5 做视频文本编码,影响外溢到视频生成</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
