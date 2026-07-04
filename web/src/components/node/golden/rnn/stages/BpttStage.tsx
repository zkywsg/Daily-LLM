import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RNN_SOURCE_PATH } from "../lib/prose";
import { BpttGradientChart } from "../widgets/BpttGradientChart";
import { MemoryTaskDemo } from "../widgets/MemoryTaskDemo";
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

export function BpttStage({ mechanism3Prose, synergyProse }: Props) {
  const [radius, setRadius] = useState(0.95);
  const [distance, setDistance] = useState(4);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:BPTT — 按时间展开后跑标准反传
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把循环按时间展开成深度 T 的前馈网络,跑标准反传就是 BPTT。
        但梯度回传要连乘 T 次 W_h·tanh'(h_t):谱半径 &lt;1 → 梯度指数衰减(梯度消失,
        长依赖学不到);谱半径 &gt;1 → 梯度指数放大(梯度爆炸,训练 NaN)。
        梯度爆炸有 gradient clipping 补丁,梯度消失是结构性问题,逼出了 LSTM 的设计。
      </p>

      <BpttGradientChart spectralRadius={radius} />
      <p className={styles.caption}>
        ↑ 拖动谱半径看梯度回传曲线。&lt;1 指数衰减,&gt;1 指数爆炸,≈1 相对稳定但很脆弱。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>W_h 谱半径</span><strong>{radius.toFixed(2)}</strong>
      </label>
      <input type="range" min={0.5} max={1.3} step={0.02} value={radius}
             onChange={(e) => setRadius(parseFloat(e.target.value))} style={{ width: "100%" }} />
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button type="button" onClick={() => setRadius(0.7)} style={btnStyle(radius === 0.7)}>0.7(消失)</button>
        <button type="button" onClick={() => setRadius(1.0)} style={btnStyle(radius === 1.0)}>1.0(临界)</button>
        <button type="button" onClick={() => setRadius(1.2)} style={btnStyle(radius === 1.2)}>1.2(爆炸)</button>
      </div>

      <MemoryTaskDemo distance={distance} />
      <p className={styles.caption}>
        ↑ 简单记忆任务:第 1 步的 "cat" 到第 T 步能否还记得。拖动看距离对记忆强度的影响。
      </p>
      <input type="range" min={1} max={5} step={1} value={distance}
             onChange={(e) => setDistance(parseInt(e.target.value))} style={{ width: "100%", marginTop: 8 }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={RNN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={RNN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              两个明确遗憾
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>梯度消失/爆炸 — LSTM(1997)用 cell highway 解决</li>
              <li>串行不可并行 — Transformer(2017)用 self-attention 彻底放弃循环解决</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
