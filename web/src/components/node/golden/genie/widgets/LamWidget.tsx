import { useState } from "react";
import { INITIAL_FRAME, ACTION_VECTORS, inferLatentAction, makeFrame } from "../lib/data";

function MiniFrame({ cx, cy }: { cx: number; cy: number }) {
  const frame = makeFrame(cx, cy);
  const size = 90, grid = 6, cell = size / grid;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {frame.map((v, i) => {
        const x = (i % grid) * cell, y = Math.floor(i / grid) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${Math.round(g * 0.6)},${g})`} />;
      })}
    </svg>
  );
}

export function LamWidget() {
  const [targetIdx, setTargetIdx] = useState(0);
  const prev = INITIAL_FRAME;
  const target = ACTION_VECTORS[targetIdx];
  const curr = { cx: prev.cx + target.dx * 0.8, cy: prev.cy + target.dy * 0.8 };
  const inferred = inferLatentAction(prev, curr);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {ACTION_VECTORS.map((a, i) => (
          <button
            key={a.label} type="button" onClick={() => setTargetIdx(i)} aria-pressed={i === targetIdx}
            style={{
              width: 30, height: 30, borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === targetIdx ? "#d946ef" : "var(--border)"}`,
              background: i === targetIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === targetIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {i}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: "var(--space-4)", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>帧 t</div>
          <MiniFrame cx={prev.cx} cy={prev.cy} />
        </div>
        <div style={{ fontSize: "var(--fs-xl)", color: "var(--ink-muted)" }}>→</div>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>帧 t+1</div>
          <MiniFrame cx={curr.cx} cy={curr.cy} />
        </div>
      </div>
      <div style={{ marginTop: "var(--space-3)", padding: "var(--space-3)", border: "1px solid #d946ef", borderRadius: "var(--radius-md)", background: "#fae8ff" }}>
        <span style={{ fontSize: "var(--fs-sm)", color: "#86198f" }}>
          LAM 无监督推断出的 latent action id = <strong>{inferred}</strong>(未使用任何人工动作标注,纯粹从两帧的差异里推断)
        </span>
      </div>
    </div>
  );
}
