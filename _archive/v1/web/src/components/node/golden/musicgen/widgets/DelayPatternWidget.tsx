import { useState } from "react";
import { NUM_LEVELS, buildDelayPattern } from "../lib/data";

const CELL = 34;

export function DelayPatternWidget() {
  const [numTimesteps, setNumTimesteps] = useState(6);
  const rows = buildDelayPattern(numTimesteps);
  const S = numTimesteps + NUM_LEVELS - 1;

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        原始时间步数 T = {numTimesteps}(交错后总长度 = T + {NUM_LEVELS} - 1 = {S})
        <input type="range" min={4} max={8} value={numTimesteps} onChange={(e) => setNumTimesteps(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${S * CELL + 80} ${NUM_LEVELS * CELL + 40}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`延迟交错模式,T=${numTimesteps}`}>
        <text x={(S * CELL + 80) / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          延迟交错模式(delay pattern):每层延迟自己的层号开始
        </text>
        {rows.map((row, level) => (
          <g key={level}>
            <text x={20} y={40 + level * CELL + CELL / 2 + 4} fontSize={10} fill="var(--ink-muted)">层{level}</text>
            {row.map((cell, s) => (
              <rect
                key={s}
                x={50 + s * CELL}
                y={30 + level * CELL}
                width={CELL - 2}
                height={CELL - 2}
                fill={cell.filled ? "#fb7185" : "var(--bg-subtle)"}
                stroke="var(--bg-canvas)"
              />
            ))}
          </g>
        ))}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        粉色格子是真实 token,灰色是 padding。每层沿对角线错开一步,让单个自回归 Transformer 按固定顺序逐帧预测,同时覆盖所有层——不需要为每层单独训练模型或加阶段。
      </p>
    </div>
  );
}
