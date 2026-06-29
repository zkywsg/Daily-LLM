import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LSTM_SOURCE_PATH } from "../lib/prose";
import { DEMO_SCENARIOS } from "../lib/lstm";
import { HiddenVsCellChannels } from "../widgets/HiddenVsCellChannels";
import { HCRoleCompare } from "../widgets/HCRoleCompare";
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

export function HiddenCellSeparationStage({ mechanism3Prose, synergyProse }: Props) {
  const [scenarioIdx, setScenarioIdx] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Hidden State 与 Cell State 分离 — 短期 vs 长期
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Vanilla RNN 只有一个 h 同时承担\"对外输出\"和\"对内传递\"两个任务,
        互相冲突。LSTM 把它拆成两条 channel:C 在内部 timestep 之间累积
        长程记忆走 highway,h = o·tanh(C) 每步重塑后对外发声。
        这样长程信号有独立通路,不被输出操作覆盖。
      </p>

      <HiddenVsCellChannels scenarioIdx={scenarioIdx} />
      <p className={styles.caption}>
        ↑ 跑 20 步 demo 序列。粉色 C_t 像\"水池\"缓慢累积;绿色 h_t
        每步被 output gate 重塑,变化更剧烈。\"周期性清零\" 场景里浅红条
        显示 forget gate ≈ 0 的步,C 直接被清空。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={LSTM_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={LSTM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <HCRoleCompare />
          <p className={styles.caption}>
            两条通道各管一面:h 对外说话,C 对内长记。output gate 是\"门户\",
            决定 C 哪些位置该露面给输出层。
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
              demo 场景
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {DEMO_SCENARIOS.map((s, i) => (
                <button key={i} type="button" onClick={() => setScenarioIdx(i)} style={btnStyle(i === scenarioIdx)}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
