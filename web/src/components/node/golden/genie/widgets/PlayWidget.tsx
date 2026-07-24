import { useState } from "react";
import { GRID_SIZE, INITIAL_FRAME, ACTION_VECTORS, dynamicsStep, makeFrame } from "../lib/data";

function FrameSvg({ cx, cy, size = 180 }: { cx: number; cy: number; size?: number }) {
  const frame = makeFrame(cx, cy);
  const cell = size / GRID_SIZE;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`当前帧,亮斑位置 (${cx.toFixed(1)}, ${cy.toFixed(1)})`}>
      {frame.map((v, i) => {
        const x = (i % GRID_SIZE) * cell, y = Math.floor(i / GRID_SIZE) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${Math.round(g * 0.6)},${g})`} stroke="var(--bg-canvas)" strokeWidth={0.5} />;
      })}
    </svg>
  );
}

// 交互式"玩":点击 8 个离散动作按钮之一,动态模型自回归生成下一帧。
// 全程只用 tokenizer/LAM 学到的离散动作空间,不需要连续控制信号。

export function PlayWidget() {
  const [pos, setPos] = useState(INITIAL_FRAME);
  const [history, setHistory] = useState<number[]>([]);

  return (
    <div>
      <FrameSvg cx={pos.cx} cy={pos.cy} />
      <div style={{ display: "flex", gap: 6, marginTop: "var(--space-3)", flexWrap: "wrap" }}>
        {ACTION_VECTORS.map((a, i) => (
          <button
            key={a.label} type="button"
            onClick={() => { setPos((p) => dynamicsStep(p, i)); setHistory((h) => [...h, i]); }}
            style={{
              width: 32, height: 32, borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)", background: "var(--bg-surface)",
              color: "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {i}
          </button>
        ))}
        <button
          type="button" onClick={() => { setPos(INITIAL_FRAME); setHistory([]); }}
          style={{ padding: "4px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-xs)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        已执行动作序列:{history.join(" → ") || "(无)"}。每点一个动作按钮,动态模型就自回归生成下一帧——这就是"用学到的离散动作逐帧玩生成出来的世界"。
      </p>
    </div>
  );
}
