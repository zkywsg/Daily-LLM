import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CONSTITUTIONAL_AI_SOURCE_PATH } from "../lib/prose";
import { HelpfulHarmlessTradeoffChart } from "../widgets/HelpfulHarmlessTradeoffChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
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

export function RlaifStage({ mechanism3Prose, synergyProse }: Props) {
  const [view, setView] = useState<"tradeoff" | "table">("tradeoff");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:RLAIF(Reinforcement Learning from AI Feedback)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        SL-CAI 模型对同一 prompt 生成两个候选回答,LLM(而非人类)按 constitution
        选出更好的一个,生成 AI 偏好数据训练 reward model,再用标准 PPO 微调 ——
        这一步彻底替代了 InstructGPT 里"40 名标注员 6 个月"的人工偏好标注。
      </p>

      {view === "tradeoff" ? (
        <HelpfulHarmlessTradeoffChart />
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "var(--fs-sm)" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid var(--border)" }}>
                <th style={{ textAlign: "left", padding: 8 }}>方法</th>
                <th style={{ textAlign: "right", padding: 8 }}>Helpfulness 胜率</th>
                <th style={{ textAlign: "right", padding: 8 }}>Harmlessness 胜率</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: 8 }}>Helpful-only RLHF baseline</td>
                <td style={{ textAlign: "right", padding: 8 }}>51%</td>
                <td style={{ textAlign: "right", padding: 8, color: "#ef4444" }}>-23%</td>
              </tr>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: 8 }}>Standard RLHF(人类反馈)</td>
                <td style={{ textAlign: "right", padding: 8 }}>50%</td>
                <td style={{ textAlign: "right", padding: 8 }}>0%(基准)</td>
              </tr>
              <tr>
                <td style={{ padding: 8, fontWeight: 700 }}>Constitutional AI(RLAIF)</td>
                <td style={{ textAlign: "right", padding: 8, fontWeight: 700 }}>51%</td>
                <td style={{ textAlign: "right", padding: 8, fontWeight: 700, color: "#10b981" }}>+9%</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
      <p className={styles.caption}>
        ↑ 切换看散点图 / 原始表格 —— Constitutional AI 在 harmlessness 上超过人类反馈 RLHF(+9%),
        helpfulness 基本持平(51% vs 50%),没有出现"变得回避/不帮忙"的失败模式。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setView("tradeoff")} style={btnStyle(view === "tradeoff")}>权衡散点图</button>
        <button type="button" onClick={() => setView("table")} style={btnStyle(view === "table")}>论文 Table 1</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,CAI 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 Constitution + RLAIF</strong>:没有 SL-CAI 起步,模型还不会拒绝有害请求,AI 偏好信号弱</li>
              <li><strong>只有 SL-CAI + RLAIF</strong>:没有书面 constitution,LLM 自评标准漂移,数据质量差</li>
              <li><strong>只有 Constitution + SL-CAI</strong>:没有 RLAIF 强化阶段,能力上限被锁在 SFT 数据质量</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
