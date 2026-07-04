import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { O1_SOURCE_PATH } from "../lib/prose";
import { HiddenTraceDiagram } from "../widgets/HiddenTraceDiagram";
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

export function HiddenTraceStage({ mechanism3Prose, synergyProse }: Props) {
  const [revealed, setRevealed] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:推理 trace 隐藏 + 长 thinking 接受度
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        thinking 过程对用户隐藏,只给 summary + answer。这背后是两个动机叠加:
        防止推理 trace(o1 真正的"秘方")被蒸馏,以及让用户更能接受"模型可以慢"
        这一新使用习惯 — 看见一行 thinking 滚 30 秒会比看见"Thought for 30s"更焦虑。
      </p>

      <HiddenTraceDiagram revealed={revealed} />
      <p className={styles.caption}>
        ↑ 点按钮切换看用户实际界面 vs 内部 thinking trace(教学演示,真实 API 不返回 reasoning 内容)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setRevealed(false)} style={btnStyle(!revealed)}>用户视角(折叠)</button>
        <button type="button" onClick={() => setRevealed(true)} style={btnStyle(revealed)}>教学视角(展开)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={O1_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={O1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,o1 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 test-time scaling + 产品化,没有 RL 内化</strong>:仍是 CoT 时代,benchmark 卡在 AIME ~13%</li>
              <li><strong>只有 RL 内化 + 产品化,没有 scaling 观察</strong>:训出长 thinking 但不会构建 o1-mini/o1/o1-pro 分层产品</li>
              <li><strong>只有 RL 内化 + scaling,没有产品化</strong>:用户体验崩溃,reasoning 模型停留在论文里</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
