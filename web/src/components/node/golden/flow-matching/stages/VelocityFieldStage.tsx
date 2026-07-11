import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLOW_MATCHING_SOURCE_PATH } from "../lib/prose";
import { VelocityFieldDiagram } from "../widgets/VelocityFieldDiagram";
import { SamplingStepsCompareChart } from "../widgets/SamplingStepsCompareChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function VelocityFieldStage({ mechanism2Prose }: Props) {
  const [mode, setMode] = useState<"fm" | "ddpm">("fm");
  const [t, setT] = useState(0.5);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二 + 机制三:速度场学习 → ODE 反向采样
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        直线路径上任意一点的速度都恒定等于 v_target = x₁ - x₀ —— 网络只需回归这一个
        常数方向,而不是 DDPM 那样在曲线上每点学不同的瞬时切线。训练好之后,采样从
        t=1(纯噪声)出发,反向积分确定性 ODE dx/dt = v_θ(x, t),不需要任何随机噪声
        注入,10-20 步就能拿到接近 DDPM 50 步的质量。
      </p>

      <VelocityFieldDiagram mode={mode} t={t} />
      <p className={styles.caption}>
        ↑ 切换看 Flow Matching(箭头处处同向,常数速度场)vs DDPM(箭头方向随位置变化)。拖动滑杆看 x_t 沿路径移动。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button type="button" onClick={() => setMode("fm")} style={btnStyle(mode === "fm")}>
          Flow Matching(常数场)
        </button>
        <button type="button" onClick={() => setMode("ddpm")} style={btnStyle(mode === "ddpm")}>
          DDPM(变化场)
        </button>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginLeft: 12 }}>
          t =
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
            style={{ width: 140 }}
          />
          {t.toFixed(2)}
        </label>
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <SamplingStepsCompareChart />
        <p className={styles.caption}>
          ↑ ODE Euler 反向采样比 SDE 反向所需步数少得多 —— 数字来自源文档"训练细节"表(SD3 默认 50 步,20 步也 work)与机制三采样代码注释。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么"少学一点"反而更好?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              DDPM 要学"曲线上每一点的瞬时切线方向",任务本身随时间变化。Flow
              Matching 把路径拉直后,整条路径上速度都是同一个常数向量
              x₁ - x₀ —— 回归目标更简单,训练更稳,采样也能大步走。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
