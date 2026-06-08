import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { scaleLinear } from "d3-scale";
import type { FamiliesData, NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { NodeHoverCard } from "./NodeHoverCard";

interface TimeAxisViewProps {
  data: FamiliesData;
}

const PADDING = 80;
const AXIS_Y = 200;
const NODE_RADIUS = 8;

export function TimeAxisView({ data }: TimeAxisViewProps) {
  const allNodes = data.families.flatMap((f) => f.nodes);
  const [hovered, setHovered] = useState<NodeData | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  if (allNodes.length === 0) {
    return <div>暂无节点数据</div>;
  }

  const minYear = Math.min(...allNodes.map((n) => n.year)) - 1;
  const maxYear = Math.max(...allNodes.map((n) => n.year)) + 1;
  const width = 1200;
  const height = 400;
  const xScale = scaleLinear()
    .domain([minYear, maxYear])
    .range([PADDING, width - PADDING]);

  // y 偏移：同一年多个节点错开
  const yByNode = new Map<string, number>();
  const groupedByYear = new Map<number, NodeData[]>();
  for (const n of allNodes) {
    if (!groupedByYear.has(n.year)) groupedByYear.set(n.year, []);
    groupedByYear.get(n.year)!.push(n);
  }
  for (const [, nodes] of groupedByYear) {
    nodes.sort((a, b) => a.family.localeCompare(b.family));
    nodes.forEach((n, i) => {
      yByNode.set(n.path, AXIS_Y - (i - (nodes.length - 1) / 2) * 24);
    });
  }

  const years: number[] = [];
  for (let y = Math.ceil(minYear); y <= Math.floor(maxYear); y += 2) {
    years.push(y);
  }

  return (
    <div style={{ position: "relative" }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", height: "auto", maxHeight: 500 }}
      >
        <line
          x1={PADDING}
          x2={width - PADDING}
          y1={AXIS_Y}
          y2={AXIS_Y}
          stroke="var(--border)"
          strokeWidth={1.5}
        />
        {years.map((y) => (
          <g key={y} transform={`translate(${xScale(y)}, ${AXIS_Y})`}>
            <line y2={6} stroke="var(--ink-muted)" />
            <text
              y={22}
              textAnchor="middle"
              fontSize={12}
              fill="var(--ink-muted)"
            >
              {y}
            </text>
          </g>
        ))}
        {allNodes.map((n) => (
          <motion.circle
            key={n.path}
            cx={xScale(n.year)}
            cy={yByNode.get(n.path)!}
            r={NODE_RADIUS}
            fill={familyColorVar(n.family)}
            stroke="var(--bg-canvas)"
            strokeWidth={2}
            whileHover={{ scale: 1.4 }}
            onMouseEnter={(e) => {
              setHovered(n);
              const target = e.currentTarget as SVGCircleElement;
              const svgEl = target.ownerSVGElement!;
              const rect = svgEl.getBoundingClientRect();
              // map SVG viewBox coord to screen
              const scaleX = rect.width / width;
              const scaleY = rect.height / height;
              setPos({
                x: rect.left + window.scrollX + xScale(n.year) * scaleX - 120,
                y: rect.top + window.scrollY + yByNode.get(n.path)! * scaleY - 200,
              });
            }}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: "pointer" }}
            layoutId={`node-${n.path}`}
          />
        ))}
      </svg>
      <AnimatePresence>
        {hovered && <NodeHoverCard node={hovered} x={pos.x} y={pos.y} />}
      </AnimatePresence>
    </div>
  );
}
