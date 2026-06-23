import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DDPM_SOURCE_PATH } from "../lib/prose";
import type { Schedule } from "../lib/math";
import { ObjectiveComparison } from "../widgets/ObjectiveComparison";
import { LossWeightCurve } from "../widgets/LossWeightCurve";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const T = 1000;

export function SimplifiedObjectiveStage({
  mechanism3Prose,
  synergyProse,
}: Props) {
  const [schedule, setSchedule] = useState<Schedule>("linear");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Simplified Objective — 把 ELBO 化简成 L2 噪声预测
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        理论上的目标是最大化 ELBO(变分下界),展开后是 T 个 KL 加权和,
        权重随 t 剧烈变化。Ho 2020 发现把所有 t 权重都拍平成 1 反而训练更稳、
        生成更好 —— 这是 DDPM 能 work 的最后一块拼图。
      </p>

      <ObjectiveComparison />
      <p className={styles.caption}>
        ↑ 左:严格 ELBO 分解(理论严谨,但梯度抖)。 右:L_simple = E‖ε - ε_θ‖²
        (实测效果反而更好)。化简的代价是丢掉了步级权重,但 Ho 在论文里
        实验证明这反而帮了 U-Net。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DDPM_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DDPM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <LossWeightCurve T={T} schedule={schedule} />
          <p className={styles.caption}>
            蓝色 = ELBO 展开后每步的隐含权重(早期 t 大、晚期掉到接近 0)。
            绿色虚线 = L_simple 把它拍平。后者均匀分配学习预算 →
            U-Net 在所有 timestep 都见过等量梯度。
          </p>
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              schedule(影响权重曲线形状)
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {(["linear", "cosine"] as Schedule[]).map((s) => {
                const active = schedule === s;
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSchedule(s)}
                    style={{
                      padding: "4px 12px",
                      fontSize: "var(--fs-sm)",
                      borderRadius: "var(--radius-sm)",
                      border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
                      background: active ? "var(--accent-link)" : "var(--bg-surface)",
                      color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {s}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
