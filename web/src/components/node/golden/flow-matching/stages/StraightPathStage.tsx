import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLOW_MATCHING_SOURCE_PATH } from "../lib/prose";
import { StraightVsCurvedPathDiagram } from "../widgets/StraightVsCurvedPathDiagram";
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

export function StraightPathStage({ intuitionProse, mechanism1Prose }: Props) {
  const [mode, setMode] = useState<"straight" | "curved">("straight");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:直线路径插值 — 替代 noise schedule
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DDPM 用 noise schedule(β_t)决定 x_t 在 noise → data 之间走一条弯曲路径,
        schedule 本身是经验调的超参。Flow Matching 反其道而行:直接定义
        x_t = (1-t)·x₀ + t·x₁ —— 一条从数据点到噪声点的直线插值,t ∈ [0,1] 均匀采,
        完全不需要任何 noise schedule。
      </p>

      <StraightVsCurvedPathDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看 Flow Matching 的直线路径 vs DDPM 的曲线随机路径 —— 同样连接 x₀ 与 x₁ 两点,路径形状完全不同。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("straight")} style={btnStyle(mode === "straight")}>
          Flow Matching(直线)
        </button>
        <button type="button" onClick={() => setMode("curved")} style={btnStyle(mode === "curved")}>
          DDPM(曲线)
        </button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              直线插值公式
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`x_t = (1 - t) · x_0 + t · x_1
t ~ U[0, 1]

v(x_t, t) = dx_t/dt = x_1 - x_0
(整条直线方向恒定)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              没有 β_t、没有 cosine schedule —— t 直接均匀采样,x_t 就是两点间的线性组合。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
