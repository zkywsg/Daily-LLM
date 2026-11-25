import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEEPSEEK_V3_SOURCE_PATH } from "../lib/prose";
import { MlaCompressionDiagram } from "../widgets/MlaCompressionDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
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

export function TrainingSystemStage({ mechanism3Prose, synergyProse }: Props) {
  const [mode, setMode] = useState<"mha" | "mla">("mha");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:训练系统三件套 — MLA + MTP + FP8
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        架构与训练稳定性解决了,算力还要能扛住 671B 模型的训练成本。MLA 把
        KV cache 压到传统 MHA 的 ~1/4,支撑长上下文推理;MTP 让每个位置多预测
        几步 token,信号更密集;FP8 全流程训练把算力再压一半 —— 三者一起,
        2048 块 H800、2.79M GPU-hours 就训完 671B 模型,成本仅 $5.6M。
      </p>

      <MlaCompressionDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看传统 MHA(每 head 独立存 K/V)vs MLA(压缩到共享低秩 latent)的 KV cache 大小对比。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("mha")} style={btnStyle(mode === "mha")}>传统 MHA</button>
        <button type="button" onClick={() => setMode("mla")} style={btnStyle(mode === "mla")}>MLA(V3)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              训练效率账
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.7, color: "var(--ink-primary)" }}>
              2048 × H800 · 2.79M GPU-hours · $5.6M 训完 671B/37B 模型
              (GPT-4 估算 100M+ GPU-hours,~$100M)。
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              FP8 训练比 BF16 快 ~2×、省 ~50% 显存,是训练效率提升 10×+ 的关键一环。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
