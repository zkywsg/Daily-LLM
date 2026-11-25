import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ROPE_SOURCE_PATH } from "../lib/prose";
import { RotationDiagram } from "../widgets/RotationDiagram";
import { PeTimelineChart } from "../widgets/PeTimelineChart";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function RotationStage({ intuitionProse, mechanism1Prose }: Props) {
  const [m, setM] = useState(3);
  const [theta, setTheta] = useState(20);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:旋转矩阵的群论结构 — 内积自然只依赖相对距离
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        原版 Transformer 把位置加在 token embedding 上,语义和位置信号纠缠。
        苏剑林 2021 反问:能不能让 attention 内积 q_m·k_n 天然只依赖相对距离 m-n?
        解法是 f(x,m) = R_m·x,R_m 是旋转矩阵。因为 R_m^T R_n = R_{"{n-m}"},
        (R_m q)·(R_n k) = q·R_{"{n-m}"}·k — 位置 m、n 消失,只剩 n-m,数学保证不需要学。
      </p>

      <RotationDiagram m={m} theta={theta} />
      <p className={styles.caption}>
        ↑ 拖动 m 看向量转多少度,拖动 θ 看旋转快慢。右侧是群论性质的代数表达。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>位置 m</span><strong>{m}</strong>
      </label>
      <input type="range" min={0} max={12} step={1} value={m} onChange={(e) => setM(parseInt(e.target.value))} style={{ width: "100%" }} />
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>频率 θ(度/位置)</span><strong>{theta}°</strong>
      </label>
      <input type="range" min={5} max={60} step={5} value={theta} onChange={(e) => setTheta(parseInt(e.target.value))} style={{ width: "100%" }} />

      <PeTimelineChart />
      <p className={styles.caption}>
        ↑ 2022 年起几乎所有主流 LLM 转向 RoPE(粉色),取代 learned absolute(灰)和 relative bias(橙)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={ROPE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={ROPE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              对比 Transformer-XL / T5
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>Transformer-XL:score 展开 4 项 + left-shift trick,复杂</li>
              <li>T5:score 加分桶偏置,简单但表达力弱</li>
              <li>RoPE:几何结构保证,不需要训练学,零额外参数</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
