import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { O1_SOURCE_PATH } from "../lib/prose";
import { ParadigmEvolutionDiagram } from "../widgets/ParadigmEvolutionDiagram";
import { RlTrainingDemo } from "../widgets/RlTrainingDemo";
import { RL_TRAINING_STAGES } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
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

export function RlTrainingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [paradigmIdx, setParadigmIdx] = useState(-1);
  const [rlStage, setRlStage] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:RL 训练让模型自然输出长 thinking 过程
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        CoT 时代 reasoning 是 inference-time 技巧,模型本身并不会主动"思考"。
        o1 在数学/代码/逻辑任务上做大规模 RL,reward 是答案对错。模型在优化答对率
        的过程中自己发现:先写长 thinking 再给答案的 trajectory reward 高得多 —
        反思、回溯、自验证都是 RL 训出来的涌现行为,不是人工设计的。
      </p>

      <ParadigmEvolutionDiagram highlightIdx={paradigmIdx} />
      <p className={styles.caption}>
        ↑ 三代 reasoning 范式(GSM8K 演示),点按钮聚焦某一代。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setParadigmIdx(-1)} style={btnStyle(paradigmIdx === -1)}>全部</button>
        {["CoT", "Self-Consistency", "o1"].map((n, i) => (
          <button key={i} type="button" onClick={() => setParadigmIdx(i)} style={btnStyle(paradigmIdx === i)}>{n}</button>
        ))}
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <RlTrainingDemo stageIdx={rlStage} />
        <p className={styles.caption}>
          ↑ 同一道 AIME 题,拖动看 RL 训练不同阶段模型输出如何从"随便猜"演化到"长 thinking + 反思自验证"。
        </p>
        <input
          type="range"
          min={0}
          max={RL_TRAINING_STAGES.length - 1}
          step={1}
          value={rlStage}
          onChange={(e) => setRlStage(parseInt(e.target.value))}
          style={{ width: "100%", marginTop: 8 }}
        />
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={O1_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={O1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              四种涌现行为
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>反思</strong> — "Wait,这个方法太复杂了"</li>
              <li><strong>回溯</strong> — 尝试方法 1 失败后主动换方法 2</li>
              <li><strong>自验证</strong> — 把答案代回原方程检查</li>
              <li><strong>路径搜索</strong> — 同一道题探索多种解法再挑</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
