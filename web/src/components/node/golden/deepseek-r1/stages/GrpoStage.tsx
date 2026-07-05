import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEEPSEEK_R1_SOURCE_PATH } from "../lib/prose";
import { GrpoGroupDiagram } from "../widgets/GrpoGroupDiagram";
import { DEMO_GROUP_REWARDS } from "../lib/data";
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

const PRESETS: Record<string, number[]> = {
  "混合对错": DEMO_GROUP_REWARDS,
  "多数答对": [1, 1, 1, 1, 1, 0, 1, 1],
  "全部答错": [0, 0, 0, 0, 0, 0, 0, 0],
};

export function GrpoStage({ mechanism2Prose }: Props) {
  const [preset, setPreset] = useState("混合对错");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:GRPO — 把 value model 干掉
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        PPO 标准做法需要 value model 估计 baseline,但 LLM 时代 value model 自身就是
        7B+ 模型,显存开销巨大且估计不准。GRPO 直接对同一 prompt 采样一组 response,
        用组内均值/方差归一化作为 advantage,完全不需要额外的 value model。
      </p>

      <GrpoGroupDiagram rewards={PRESETS[preset]} />
      <p className={styles.caption}>
        ↑ 切换不同 reward 分布,观察组内归一化后 advantage(蓝=正,橙=负)如何变化 — 全部答错时 advantage 全为 0(无有效学习信号)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {Object.keys(PRESETS).map((p) => (
          <button key={p} type="button" onClick={() => setPreset(p)} style={btnStyle(preset === p)}>{p}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              GRPO vs PPO
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>省内存</strong> — 不用维护 7B+ 的 value model</li>
              <li><strong>更稳定</strong> — group baseline 对绝对 reward 噪声鲁棒</li>
              <li><strong>更适合稀疏 reward</strong> — 组内对比天然处理"只在最后给分"</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
