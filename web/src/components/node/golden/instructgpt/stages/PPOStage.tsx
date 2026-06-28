import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { INSTRUCTGPT_SOURCE_PATH } from "../lib/prose";
import { KL_CURVE } from "../lib/data";
import { PPOLoopFlow } from "../widgets/PPOLoopFlow";
import { KLConstraintCurve } from "../widgets/KLConstraintCurve";
import { AlignmentTaxBars } from "../widgets/AlignmentTaxBars";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
  alignmentTaxProse: string;
}

export function PPOStage({ mechanism3Prose, synergyProse, alignmentTaxProse }: Props) {
  const [betaIdx, setBetaIdx] = useState(2);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:PPO RL — 用 RM 当 reward fine-tune,KL 防漂移
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        把 SFT 模型当作初始 policy,用 RM 给它生成的输出打分,PPO 梯度
        最大化期望 reward。加一项 KL[π_θ || π_SFT] 约束,不让 policy 跑得太远
        —— 防 reward hacking 把语言模型玩成胡言乱语机器。
      </p>

      <PPOLoopFlow />
      <p className={styles.caption}>
        ↑ 循环:prompt → π_θ 采样 response → RM 打分 → PPO 梯度更新 π_θ。
        SFT 模型 frozen,只用于 KL 约束(虚线)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>

          {alignmentTaxProse && (
            <>
              <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
                Alignment Tax
              </h3>
              <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
                <MarkdownRenderer markdown={alignmentTaxProse} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
              </div>
            </>
          )}
        </div>

        <div className={styles.stickyPanel}>
          <KLConstraintCurve betaIdx={betaIdx} />
          <p className={styles.caption}>
            拖 β slider 看 KL 系数对 reward 和 policy drift 的权衡。
            β 太小 → reward hacking(红色危险区);β 太大 → policy 跟 SFT 一样,没收益。
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
              <span>KL 系数 β 档位</span>
              <strong>{KL_CURVE[betaIdx].beta} {KL_CURVE[betaIdx].hacked ? "⚠ hacked" : "✓ stable"}</strong>
            </label>
            <input
              type="range"
              min={0}
              max={KL_CURVE.length - 1}
              step={1}
              value={betaIdx}
              onChange={(e) => setBetaIdx(parseInt(e.target.value, 10))}
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
              原论文 β ≈ 0.02 — 在曲线左侧第二个点附近,balance 之处。
            </div>
          </div>

          <div style={{ marginTop: "var(--space-6)" }}>
            <AlignmentTaxBars />
          </div>
        </div>
      </div>
    </div>
  );
}
