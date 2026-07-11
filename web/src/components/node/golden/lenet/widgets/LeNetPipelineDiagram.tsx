import { useState } from "react";
import { LENET_LAYERS } from "../lib/data";

const W = 760;
const H = 260;

const KIND_COLOR: Record<string, { fill: string; stroke: string }> = {
  input: { fill: "#fef3c7", stroke: "#f59e0b" },
  conv: { fill: "#fce7f3", stroke: "#ec4899" },
  pool: { fill: "#dbeafe", stroke: "#3b82f6" },
  fc: { fill: "#ecfdf5", stroke: "#10b981" },
  output: { fill: "#ecfdf5", stroke: "#10b981" },
};

export function LeNetPipelineDiagram() {
  const [active, setActive] = useState(1);

  const n = LENET_LAYERS.length;
  const colW = 70;
  const gap = (W - 60 - n * colW) / (n - 1);
  const startX = 30;
  const rowY = 120;
  const maxChannels = Math.max(...LENET_LAYERS.map((l) => l.channels));

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
        role="img"
        aria-label="LeNet-5 主干流水线 C1 到 output 的形状变化"
      >
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          LeNet-5 主干:C1 → S2 → C3 → S4 → C5 → F6 → output
        </text>

        {LENET_LAYERS.map((layer, i) => {
          const x = startX + i * (colW + gap);
          const h = 24 + (layer.channels / maxChannels) * 90;
          const c = KIND_COLOR[layer.kind];
          const isActive = i === active;
          return (
            <g
              key={layer.label}
              onClick={() => setActive(i)}
              style={{ cursor: "pointer" }}
            >
              <rect
                x={x}
                y={rowY - h / 2}
                width={colW}
                height={h}
                fill={c.fill}
                stroke={c.stroke}
                strokeWidth={isActive ? 3 : 1.4}
                rx={4}
                opacity={isActive ? 1 : 0.75}
              />
              <text x={x + colW / 2} y={rowY - h / 2 - 22} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
                {layer.label}
              </text>
              <text x={x + colW / 2} y={rowY - h / 2 - 8} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">
                {layer.size}
              </text>
              <text x={x + colW / 2} y={rowY + h / 2 + 16} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
                {layer.channels}ch
              </text>
              {i < n - 1 && (
                <line
                  x1={x + colW}
                  y1={rowY}
                  x2={x + colW + gap}
                  y2={rowY}
                  stroke="var(--border)"
                  strokeWidth={1.5}
                  markerEnd="url(#arrow-lenet-pipeline)"
                />
              )}
            </g>
          );
        })}

        <defs>
          <marker id="arrow-lenet-pipeline" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--border)" />
          </marker>
        </defs>
      </svg>

      <div
        style={{
          padding: "var(--space-3) var(--space-4)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          background: "var(--bg-surface)",
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
        }}
      >
        <strong style={{ color: "var(--ink-primary)" }}>{LENET_LAYERS[active].label}</strong>
        {"  "}
        {LENET_LAYERS[active].note}
      </div>

      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        {LENET_LAYERS.map((layer, i) => (
          <button
            key={layer.label}
            type="button"
            onClick={() => setActive(i)}
            style={{
              padding: "3px 8px",
              fontSize: "var(--fs-xs)",
              borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === active ? "var(--accent-link)" : "var(--border)"}`,
              background: i === active ? "var(--accent-link)" : "var(--bg-surface)",
              color: i === active ? "var(--bg-surface)" : "var(--ink-secondary)",
              cursor: "pointer",
            }}
          >
            {layer.label}
          </button>
        ))}
      </div>
    </div>
  );
}
