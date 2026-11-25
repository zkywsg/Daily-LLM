import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { REACT_SOURCE_PATH } from "../lib/prose";
import { ParadigmCompareChart } from "../widgets/ParadigmCompareChart";
import { PARADIGM_COMPARE } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function ObservationStage({ mechanism3Prose, synergyProse }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Observation — 让结果回喂闭环
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Observation 是 ReAct 闭环的"眼"。每次 Action 的结果直接拼回 prompt
        作为新上下文。下一轮 Thought 看到 Observation 后,才能 ground 出
        下一步该做什么。Observation 让闭环真正闭合 — 少了它,Action 调完
        也不知道结果,等于白调。
      </p>

      <ParadigmCompareChart highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 五种范式在 HotpotQA / Fever 上的对比,点按钮聚焦某一行 — ReAct 全面领先。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {PARADIGM_COMPARE.map((p, i) => (
          <button key={p.pattern} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{p.pattern}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={REACT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={REACT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,闭环都启动不了
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 Thought(=CoT)</strong>:全程脑内推理,HotpotQA 30.6,容易瞎编</li>
              <li><strong>只有 Action(=Act-only)</strong>:能查工具但没推理目的,HotpotQA 25.7,不知何时停</li>
              <li><strong>只有 Observation</strong>:没人决定调什么也没人解释为什么,闭环没驱动力</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
