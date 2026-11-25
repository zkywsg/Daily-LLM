import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LEARNING_TO_SUMMARIZE_SOURCE_PATH } from "../lib/prose";
import { KL_TRADEOFF } from "../lib/data";
import { PpoFineTuneDiagram } from "../widgets/PpoFineTuneDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PpoStage({ mechanism3Prose, synergyProse }: Props) {
  const [betaIdx, setBetaIdx] = useState(1);
  const row = KL_TRADEOFF[betaIdx];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:PPO + KL 惩罚 — 双重防 reward hacking
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把 RM 当 reward 函数,用 PPO 优化 policy:max E[r(x,y)] − β·KL(π_θ‖π_SFT)。
        第一项让模型追求高分,第二项(KL 惩罚)把 policy 锚在 SFT 附近 —— 没有它,
        模型会学会"骗 reward model 打高分"的乱码输出,这就是 reward hacking。
      </p>

      <PpoFineTuneDiagram betaIdx={betaIdx} />
      <p className={styles.caption}>
        ↑ 拖动 β 档位看 reward 和 KL 散度的权衡:β 太小 reward hacking 风险高,β 太大收益趋近于零。
      </p>
      <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)", maxWidth: 420 }}>
        <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
          <span>KL 系数 β 档位</span>
          <strong>{row.beta} {row.hacked ? "⚠ hacked" : "✓ stable"}</strong>
        </label>
        <input
          type="range"
          min={0}
          max={KL_TRADEOFF.length - 1}
          step={1}
          value={betaIdx}
          onChange={(e) => setBetaIdx(parseInt(e.target.value, 10))}
          style={{ width: "100%" }}
        />
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 4, lineHeight: 1.4 }}>
          论文典型 β = 0.01–0.1,自适应调整(KL 超过目标值则增大 β)。
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              PPO 优化目标
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>
{`max E[r(x,y)] - β·KL(π_θ‖π_SFT)

r(x,y) = RM 打分(想要高)
KL 惩罚 = 防止偏离 SFT 太远
4 个模型同显存:
  actor / critic / RM / ref`}
            </pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              PPO 在 LLM 上是工程恶梦:显存爆、训练周级、调参难 —— 这是 2023 年 DPO 出来后社区转向 DPO 的根本原因。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
