import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT3_SOURCE_PATH } from "../lib/prose";
import { fmtParams } from "../lib/scaling";
import { KaplanScalingCurve } from "../widgets/KaplanScalingCurve";
import { ModelLineupBar } from "../widgets/ModelLineupBar";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ScalingLawStage({ intuitionProse, mechanism1Prose }: Props) {
  const [logParams, setLogParams] = useState(11.24); // ≈ 175B

  const params = Math.pow(10, logParams);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:175B 参数 — 沿 Kaplan scaling law 直接外推
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Kaplan et al. 2020 发现:test loss 对参数量 N 服从 power law。
        OpenAI 据此直接外推 —— 不调架构、不改算法,只把 GPT-2 的 1.5B 放大 117×
        到 175B。结果是 in-context learning、few-shot 能力自然涌现。
      </p>

      <KaplanScalingCurve currentParams={params} />
      <p className={styles.caption}>
        ↑ 拖下面的 slider 看不同参数量在 Kaplan 曲线上的位置 + 预测 loss。
        红色曲线是拟合,彩色点是已发布模型,绿色点是当前 slider。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GPT3_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GPT3_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ModelLineupBar />
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
              <span>参数量 N (log10)</span>
              <strong>{fmtParams(params)}</strong>
            </label>
            <input
              type="range"
              min={7}
              max={13}
              step={0.1}
              value={logParams}
              onChange={(e) => setLogParams(parseFloat(e.target.value))}
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
              拖到 117M(GPT-1)→ 1.5B(GPT-2)→ 175B(GPT-3)→ 1.8T(GPT-4 估)
              看曲线上的 loss 预测。注意是 log 横轴 —— 每跨一格是 10×。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
