import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { IMAGEN_SOURCE_PATH } from "../lib/prose";
import { CascadePipeline } from "../widgets/CascadePipeline";
import { CascadeVsLdm } from "../widgets/CascadeVsLdm";
import { CASCADE_STAGES } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function CascadeStage({ mechanism2Prose }: Props) {
  const [stageIdx, setStageIdx] = useState(-1);
  const [side, setSide] = useState<"cascade" | "ldm" | "both">("both");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:三级 Cascade Diffusion — 分辨率逐级提升
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        三个独立 diffusion 模型分别做不同分辨率:text → 64×64 用大模型学语义映射,
        后两阶段是 super-resolution,只学高频细节,不需要重新学文本理解,模型可以小。
        对比 LDM 的"VAE 压缩 + 一次性 latent diffusion",Cascade 在 pixel 空间做 3 次,
        没有 VAE 信息损失但工程更复杂。
      </p>

      <CascadePipeline highlightIdx={stageIdx} />
      <p className={styles.caption}>
        ↑ 点按钮聚焦某一阶段。第一阶段(64×64)承担所有语义理解,后两阶段只是"放大 + 补细节"。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button type="button" onClick={() => setStageIdx(-1)} style={btnStyle(stageIdx === -1)}>全部</button>
        {CASCADE_STAGES.map((s, i) => (
          <button key={i} type="button" onClick={() => setStageIdx(i)} style={btnStyle(stageIdx === i)}>{s.name}</button>
        ))}
      </div>

      <CascadeVsLdm side={side} />
      <p className={styles.caption}>
        ↑ Imagen 选 cascade 是因为 Google 不缺算力;社区选 LDM 是因为要塞进消费 GPU。两条路并存至今。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("cascade")} style={btnStyle(side === "cascade")}>聚焦 Cascade</button>
        <button type="button" onClick={() => setSide("ldm")} style={btnStyle(side === "ldm")}>聚焦 LDM</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={IMAGEN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              两条工业路线
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>Google 系</strong>:Imagen / Imagen Video / Lumiere → cascade</li>
              <li><strong>社区系</strong>:SD / Midjourney / Flux → LDM latent</li>
              <li>Cascade 质量可能略好(无 VAE 损失)但总算力更大</li>
              <li>LDM 塞进消费 GPU,是开源生态可行的关键</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
