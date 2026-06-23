import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DDPM_SOURCE_PATH } from "../lib/prose";
import type { Schedule } from "../lib/math";
import { ReverseStrip } from "../widgets/ReverseStrip";
import { EpsilonPredictionView } from "../widgets/EpsilonPredictionView";
import { ReverseControls } from "../widgets/ReverseControls";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const T = 1000;
const SCHEDULE: Schedule = "linear";

export function ReverseProcessStage({ mechanism2Prose }: Props) {
  const [t, setT_t] = useState(400);
  const [epsilonNoise, setEpsilonNoise] = useState(0.3);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Reverse Process — 学一个 ε_θ 去预测每步噪声
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Forward 的真后验 q(x_{"ₜ₋₁"} | x_t, x_0) 是高斯,但实际采样时
        x_0 未知。DDPM 训练一个 U-Net ε_θ(x_t, t) 预测当前步加进去的噪声,
        然后用 close-form 拼出 μ_{"ₜ₋₁"} —— 一步一步把 x_T 推回 x_0。
      </p>

      <EpsilonPredictionView
        T={T}
        t={t}
        schedule={SCHEDULE}
        epsilonNoise={epsilonNoise}
      />
      <p className={styles.caption}>
        ↑ 训练目标:让虚线(网络预测 ε_θ)对齐实线(真噪声 ε)。
        δ 越小 MSE 越小 —— 这正是 L_simple 的优化方向。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={mechanism2Prose}
              sourcePath={DDPM_SOURCE_PATH}
            />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ReverseStrip
            T={T}
            t={t}
            schedule={SCHEDULE}
            epsilonNoise={epsilonNoise}
          />
          <p className={styles.caption}>
            绿色 μ_{"ₜ₋₁"} 是反向一步的估计。δ 趋近 0 时它紧贴 x_0
            (虚线);δ 大时偏离 —— 这就是为啥 ε_θ 没训好,采样会糊。
          </p>
          <div style={{ marginTop: "var(--space-4)" }}>
            <ReverseControls
              t={t}
              onTimeChange={setT_t}
              T={T}
              epsilonNoise={epsilonNoise}
              onEpsilonNoiseChange={setEpsilonNoise}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
