import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LSTM_SOURCE_PATH } from "../lib/prose";
import { CellHighwayDiagram } from "../widgets/CellHighwayDiagram";
import { GradientDecayCurve } from "../widgets/GradientDecayCurve";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function CellHighwayStage({ intuitionProse, mechanism1Prose }: Props) {
  const [forgetMean, setForgetMean] = useState(0.95);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Cell State 高速路 — 梯度不被反复乘
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Vanilla RNN 反向传播梯度 ~ Π (W · f'(z)),T 步后基本归零 ——
        长程依赖学不到。LSTM 在 cell state 上开一条 \"高速路\":
        C_t = f_t ⊙ C_{"ₜ₋₁"} + i_t ⊙ g_t,只有 element-wise 操作。
        梯度反向沿这条路只乘 f_t(≈0.95),近似恒等,长程信号能传回去。
      </p>

      <CellHighwayDiagram />
      <p className={styles.caption}>
        ↑ 粉色横线是 cell state 高速路:只有 ⊙ f_t 和 + i_t·g_t,没有矩阵 W。
        反向传播时梯度只乘小的 f_t 标量,不被矩阵反复挤压。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={LSTM_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={LSTM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <GradientDecayCurve forgetMean={forgetMean} />
          <p className={styles.caption}>
            横轴 = 反向传播 timestep。RNN(灰)经 20 步已经在 1e-6 以下;
            LSTM(粉,forget=0.95)走 50 步还在 1e-1 量级。拖 slider
            把 forget mean 拉低到 0.6 看 LSTM 也会失效 —— forget gate
            训练时被强迫学到接近 1 是关键。
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
            <label
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                marginBottom: 4,
              }}
            >
              <span>forget gate 平均值</span>
              <strong>{forgetMean.toFixed(2)}</strong>
            </label>
            <input
              type="range"
              min={0.5}
              max={0.999}
              step={0.01}
              value={forgetMean}
              onChange={(e) => setForgetMean(parseFloat(e.target.value))}
              style={{ width: "100%" }}
            />
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginTop: 4,
                lineHeight: 1.4,
              }}
            >
              真正训练好的 LSTM forget gate 通常 0.9-1.0(初始 bias=1 让它默认\"记住\")。
              0.6 以下 LSTM 退化成 vanilla RNN。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
