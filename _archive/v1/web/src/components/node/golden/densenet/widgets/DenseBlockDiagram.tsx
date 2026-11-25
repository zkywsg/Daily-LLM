import { useState } from "react";
import { DENSE_BLOCK_NODES, cumulativeChannels, GROWTH_RATE_K } from "../lib/data";

interface Props {
  width?: number;
  height?: number;
}

/**
 * Dense block 内部连接示意:第 ℓ 层(节点)接收前面 0..ℓ-1 所有层输出的 concat。
 * 悬停任意节点高亮它接收的全部输入弧线,并显示累积输入通道数。
 */
export function DenseBlockDiagram({ width = 620, height = 340 }: Props) {
  const nodes = DENSE_BLOCK_NODES;
  const n = nodes.length;
  const margin = 60;
  const usable = width - margin * 2;
  const xs = nodes.map((_, i) => margin + (usable * i) / (n - 1));
  const nodeY = 220;
  const R = 26;

  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  // 每对 (i, j) i<j 的弧线:表示 j 接收 i 的输出
  const arcs: Array<{ from: number; to: number }> = [];
  for (let j = 1; j < n; j++) {
    for (let i = 0; i < j; i++) {
      arcs.push({ from: i, to: j });
    }
  }

  const arcHeight = (from: number, to: number) => 40 + (to - from) * 22;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="Dense block 稠密连接示意图"
    >
      <text x={width / 2} y={26} textAnchor="middle" fontSize={16} fontWeight={600} fill="var(--ink-primary)">
        Dense Block(5 层示意)—— 每层接收前面所有层的 concat
      </text>

      {/* 弧线 */}
      {arcs.map(({ from, to }, i) => {
        const active = hoverIdx === to;
        const h = arcHeight(from, to);
        const x1 = xs[from];
        const x2 = xs[to];
        const midX = (x1 + x2) / 2;
        const topY = nodeY - h;
        const path = `M ${x1} ${nodeY - R} Q ${midX} ${topY}, ${x2} ${nodeY - R}`;
        return (
          <path
            key={i}
            d={path}
            fill="none"
            stroke={active ? "#3b82f6" : "#dbeafe"}
            strokeWidth={active ? 2.5 : 1.5}
            strokeDasharray={active ? undefined : "3 3"}
            opacity={hoverIdx === null || active ? 1 : 0.25}
          />
        );
      })}

      {/* 节点 */}
      {nodes.map((node, i) => {
        const isInput = i === 0;
        const isHover = hoverIdx === i;
        const fill = isInput ? "#fef3c7" : "#fce7f3";
        const stroke = isInput ? "#f59e0b" : "#ec4899";
        return (
          <g
            key={node.index}
            onMouseEnter={() => setHoverIdx(i)}
            onMouseLeave={() => setHoverIdx(null)}
            style={{ cursor: "help" }}
          >
            <circle
              cx={xs[i]}
              cy={nodeY}
              r={R}
              fill={fill}
              stroke={stroke}
              strokeWidth={isHover ? 3 : 1.5}
            />
            <text x={xs[i]} y={nodeY + 5} textAnchor="middle" fontSize={13} fontWeight={600} fill={isInput ? "#92400e" : "#9d174d"}>
              {node.label}
            </text>
            <text x={xs[i]} y={nodeY + R + 20} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
              +{node.channels}ch
            </text>
          </g>
        );
      })}

      {/* 输出箭头到 block 末尾 */}
      <line
        x1={xs[n - 1] + R}
        y1={nodeY}
        x2={xs[n - 1] + R + 24}
        y2={nodeY}
        stroke="#10b981"
        strokeWidth={2}
        markerEnd="url(#dn-block-arrow)"
      />
      <defs>
        <marker id="dn-block-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#10b981" />
        </marker>
      </defs>

      {/* 悬停信息面板 */}
      <g transform={`translate(${width / 2}, ${height - 46})`}>
        <rect x={-220} y={-28} width={440} height={44} rx={6} fill="var(--bg-surface)" stroke="var(--border)" strokeWidth={1} />
        <text x={0} y={-8} textAnchor="middle" fontSize={12} fill="var(--ink-secondary)">
          {hoverIdx === null
            ? `悬停任意层查看它接收的累积输入通道数(growth rate k=${GROWTH_RATE_K})`
            : hoverIdx === 0
              ? `x₀:block 输入,${cumulativeChannels(0)} 通道`
              : `${nodes[hoverIdx].label} 的输入 = 前 ${hoverIdx} 层 concat = ${cumulativeChannels(hoverIdx)} 通道`}
        </text>
        <text x={0} y={10} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--ink-primary)">
          {hoverIdx !== null && hoverIdx > 0
            ? `x_${hoverIdx} = H_${hoverIdx}([x_0, x_1, …, x_${hoverIdx - 1}])`
            : "x_ℓ = H_ℓ([x_0, x_1, …, x_{ℓ-1}])"}
        </text>
      </g>
    </svg>
  );
}
