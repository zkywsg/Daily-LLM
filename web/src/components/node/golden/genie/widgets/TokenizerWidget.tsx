import { useState } from "react";
import { GRID_SIZE, INITIAL_FRAME, makeFrame, tokenizeFrame } from "../lib/data";

function GridSvg({ values, size = 140 }: { values: number[]; size?: number }) {
  const cell = size / GRID_SIZE;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {values.map((v, i) => {
        const x = (i % GRID_SIZE) * cell;
        const y = Math.floor(i / GRID_SIZE) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${Math.round(g * 0.6)},${g})`} stroke="var(--bg-canvas)" strokeWidth={0.5} />;
      })}
    </svg>
  );
}

export function TokenizerWidget() {
  const [cx, setCx] = useState(INITIAL_FRAME.cx);
  const frame = makeFrame(cx, INITIAL_FRAME.cy);
  const token = tokenizeFrame(frame);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        移动亮斑位置(模拟不同帧)
        <input type="range" min={0} max={5} step={0.5} value={cx} onChange={(e) => setCx(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", gap: "var(--space-4)", alignItems: "center" }}>
        <GridSvg values={frame} />
        <div style={{ fontSize: "var(--fs-2xl)", color: "var(--ink-muted)" }}>→</div>
        <div style={{ padding: "var(--space-4)", border: "1px solid #d946ef", borderRadius: "var(--radius-md)", background: "#fae8ff" }}>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>token id</div>
          <div style={{ fontSize: "var(--fs-2xl)", fontWeight: 700, color: "#86198f" }}>{token}</div>
        </div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        tokenizer 把每一帧压缩成一个离散 token,后续 LAM 和动态模型全部在 token 序列上工作,不直接处理像素。
      </p>
    </div>
  );
}
