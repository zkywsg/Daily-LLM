import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LSTM_SOURCE_PATH } from "../lib/prose";
import { ThreeGatesDiagram } from "../widgets/ThreeGatesDiagram";
import { GateInteractive } from "../widgets/GateInteractive";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

function SliderRow({
  label, value, onChange, min = 0, max = 1, step = 0.05, hue,
}: {
  label: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; hue: number;
}) {
  return (
    <div>
      <label
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          marginBottom: 4,
        }}
      >
        <span style={{ color: `hsl(${hue}, 60%, 45%)`, fontWeight: 600 }}>{label}</span>
        <strong>{value.toFixed(2)}</strong>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%" }}
      />
    </div>
  );
}

export function ThreeGatesStage({ mechanism2Prose }: Props) {
  const [f, setF] = useState(0.9);
  const [iVal, setI] = useState(0.6);
  const [o, setO] = useState(0.7);
  const [g, setG] = useState(0.5);
  const [prevC, setPrevC] = useState(0.4);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:三门控制 — forget / input / output
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        三个门各管一件事:forget 决定保留多少旧 C(0=全忘 / 1=全留),
        input 决定写入多少新 g(0=不写 / 1=全写),output 决定暴露多少
        cell state 给 hidden(0=不输出 / 1=全暴露)。配 candidate g_t 用
        tanh 给 -1..1 范围,这套四件套让 LSTM 既能记又能忘还能选择性输出。
      </p>

      <ThreeGatesDiagram />
      <p className={styles.caption}>
        ↑ 完整 LSTM cell:左侧 [h_{"ₜ₋₁"}; x_t] 分发到 4 个线性变换,
        分别生成 f/i/o(σ)和 g(tanh)。然后 C 走高速路加更新,h 由 o·tanh(C) 抽取。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={LSTM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <GateInteractive f={f} i={iVal} o={o} g={g} prevC={prevC} />
          <p className={styles.caption}>
            拖左侧 5 个 slider 实时看 newC / newH 怎么算。试试:
            f→0 立即把 cell 清零;i→0 让新输入完全无效;o→0 hidden 输出归零但 cell 还在。
          </p>
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-3)",
            }}
          >
            <SliderRow label="prevC (上一步 cell)" value={prevC} onChange={setPrevC} min={-1} max={1} step={0.05} hue={330} />
            <SliderRow label="f forget" value={f} onChange={setF} hue={30} />
            <SliderRow label="i input" value={iVal} onChange={setI} hue={330} />
            <SliderRow label="g candidate" value={g} onChange={setG} min={-1} max={1} step={0.05} hue={220} />
            <SliderRow label="o output" value={o} onChange={setO} hue={160} />
          </div>
        </div>
      </div>
    </div>
  );
}
