import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { REACT_SOURCE_PATH } from "../lib/prose";
import { ReactTraceDiagram } from "../widgets/ReactTraceDiagram";
import { CotVsReactDiagram } from "../widgets/CotVsReactDiagram";
import { REACT_TRACE } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ThoughtStage({ intuitionProse, mechanism1Prose }: Props) {
  const [step, setStep] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Thought — 让推理目的显式化
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Thought 是 ReAct 闭环的"大脑"。每步开头 LLM 必须显式写出"为什么要做
        这一步"。没有 Thought 的"Act-only"模式里,LLM 直接给 Action,不解释
        目的 — 结果就是 token 一旦多了,LLM 不知道自己在干什么,经常重复
        或跑偏。Thought 把"目的"显式落到上下文里,后续步骤可以 ground 在
        前面的推理上。
      </p>

      <ReactTraceDiagram step={step} />
      <p className={styles.caption}>
        ↑ 拖动查看 Aurora Borealis 问题的完整 Thought → Action → Observation 循环。
      </p>
      <input
        type="range"
        min={0}
        max={REACT_TRACE.length - 1}
        step={1}
        value={step}
        onChange={(e) => setStep(parseInt(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div style={{ marginTop: "var(--space-8)" }}>
        <CotVsReactDiagram />
        <p className={styles.caption}>
          ↑ 纯 CoT 容易把"氧原子"记错成"氢原子";ReAct 每步基于真实查证信息,幻觉显著下降。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={REACT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={REACT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              两条极端路线
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>CoT 闭门造车</strong> — 全程脑内推理,无法接入外部信息,容易幻觉</li>
              <li><strong>Act-only</strong> — 给一个工具调用就直接出答案,不知何时停</li>
              <li><strong>ReAct</strong> — 思 → 做 → 观 → 思,形成开放循环</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
