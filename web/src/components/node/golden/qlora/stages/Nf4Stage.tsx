import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { QLORA_SOURCE_PATH } from "../lib/prose";
import { MemoryStackDiagram } from "../widgets/MemoryStackDiagram";
import { Nf4QuantDiagram } from "../widgets/Nf4QuantDiagram";
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

export function Nf4Stage({ intuitionProse, mechanism1Prose }: Props) {
  const [mode, setMode] = useState<"nf4" | "int4">("nf4");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:NF4(NormalFloat 4-bit)量化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        LoRA 解决了训练参数问题,但 base 模型仍要以 fp16 加载 — LLaMA-65B 光 base
        就 130GB。标准 INT4 量化用均匀分布位点,但神经网络权重是正态分布,大量
        权重挤在 0 附近却只分到和两端一样多的量化级别。NF4 把 16 个量化位点选在
        正态分布的等分位点上,比 INT4 误差小 ~30%。
      </p>

      <MemoryStackDiagram />
      <p className={styles.caption}>
        ↑ LLaMA-65B 显存账:fp16 LoRA 需要 150GB(2×A100-80GB),QLoRA 只需 41GB(单卡 A6000)。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <Nf4QuantDiagram mode={mode} />
        <p className={styles.caption}>
          ↑ 切换看 NF4 vs INT4 的量化位点如何分布在权重的正态分布曲线上。
        </p>
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <button type="button" onClick={() => setMode("nf4")} style={btnStyle(mode === "nf4")}>NF4(正态分布位点)</button>
          <button type="button" onClick={() => setMode("int4")} style={btnStyle(mode === "int4")}>INT4(均匀位点)</button>
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={QLORA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={QLORA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              NF4 的 16 个值
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`[-1.0, -0.696, -0.525, -0.395,
 -0.284, -0.185, -0.091, 0.0,
 0.080, 0.161, 0.246, 0.338,
 0.441, 0.563, 0.723, 1.0]`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              0 附近位点密集(权重密集区精度高),两端稀疏(权重罕见区不浪费量化级别)。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
