import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DPO_SOURCE_PATH } from "../lib/prose";
import { RewardTrajectories } from "../widgets/RewardTrajectories";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const PRESETS: Array<{ label: string; beta: number; lr: number; note: string }> = [
  { label: "典型 (β=0.1, lr=5e-7)",     beta: 0.10, lr: 5e-7, note: "DPO 论文标配 / 大多数开源对齐用这套" },
  { label: "保守 (β=0.5, lr=5e-7)",     beta: 0.50, lr: 5e-7, note: "更被 π_ref 拉住,变化慢但安全" },
  { label: "激进 (β=0.05, lr=1e-6)",    beta: 0.05, lr: 1e-6, note: "快速对齐 chosen,易过拟合(DPO 版 reward hacking)" },
  { label: "过激进 (β=0.01, lr=2e-6)",  beta: 0.01, lr: 2e-6, note: "几乎抛弃 π_ref · catastrophic forgetting 风险" },
];

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function TrainingDynamicsStage({ mechanism3Prose, synergyProse }: Props) {
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = PRESETS[presetIdx];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:训练 = 提 chosen + 压 rejected + π_ref 锚定
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DPO loss 的梯度形式很直观:每 step 推高 π_θ(y_w) / 压低 π_θ(y_l),
        但两个方向都被 π_ref 通过 log-ratio 锚定。β 是唯一旋钮 — 等价于 PPO 的 KL 系数。
        β 大严格遵从 π_ref(慢但安全);β 小激进偏向 chosen(快但易过拟合)。
      </p>

      <RewardTrajectories beta={preset.beta} lr={preset.lr} />
      <p className={styles.caption}>
        ↑ 切换不同 β / lr 预设看 chosen vs rejected reward 在 100 step 内的走势。
        过激进配置下 reward 离 baseline 跑得太远 — 这就是 DPO 版 reward hacking。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DPO_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DPO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              超参预设
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {PRESETS.map((p, i) => (
                <button key={i} type="button" onClick={() => setPresetIdx(i)} style={btnStyle(i === presetIdx)}>
                  {p.label}
                </button>
              ))}
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              {preset.note}
            </div>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              实践经验
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>lr 是 DPO 最敏感的 hyperparameter,5e-7 是 Zephyr/Tulu 默认</li>
              <li>β 大多任务 0.1 OK,数学/代码类可试 0.01-0.05</li>
              <li>偏好数据质量比 RM 时代更关键 — 没有 RM 缓冲</li>
              <li>chosen reward 持续 ↑ 同时 rejected reward 也 ↑ → 学习失败(模型在记忆而不是对齐)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
