import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router";
import type { FamiliesData, NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { NodeHoverCard } from "./NodeHoverCard";

interface TimeAxisViewProps {
  data: FamiliesData;
}

const PADDING = 80;
const AXIS_Y = 200;
const NODE_RADIUS = 8;

// 空白年份的权重(给极小占位空间,压缩"前疏后密")
const EMPTY_YEAR_WEIGHT = 0.18;
// 有节点年份的权重函数:sqrt(count + 0.5),拉开密集 vs 稀疏年份
const yearWeight = (count: number) =>
  count === 0 ? EMPTY_YEAR_WEIGHT : Math.sqrt(count + 0.5);

export function TimeAxisView({ data }: TimeAxisViewProps) {
  const allNodes = data.families.flatMap((f) => f.nodes);
  const [hovered, setHovered] = useState<NodeData | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const navigate = useNavigate();

  if (allNodes.length === 0) {
    return <div>暂无节点数据</div>;
  }

  const minYear = Math.min(...allNodes.map((n) => n.year)) - 1;
  const maxYear = Math.max(...allNodes.map((n) => n.year)) + 1;
  const width = 1200;
  const height = 400;

  // y 偏移:同一年多个节点错开
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

  // 密度感知 x 刻度:按"每年节点数"加权分配横向空间
  // —— 空白年份只占极小宽度,密集年份拿到大头
  const yearCenters = new Map<number, number>();
  const yearWeights: number[] = [];
  for (let y = minYear; y <= maxYear; y++) {
    yearWeights.push(yearWeight(groupedByYear.get(y)?.length ?? 0));
  }
  const totalWeight = yearWeights.reduce((a, b) => a + b, 0);
  const usableWidth = width - 2 * PADDING;
  let cumulative = 0;
  for (let i = 0; i < yearWeights.length; i++) {
    const year = minYear + i;
    const w = yearWeights[i];
    yearCenters.set(
      year,
      PADDING + ((cumulative + w / 2) / totalWeight) * usableWidth,
    );
    cumulative += w;
  }
  const xScale = (year: number) => yearCenters.get(year) ?? PADDING;

  // 刻度标签:只展示有节点的年份(避免空白年份挤出一堆无用刻度)
  const labelYears = [...groupedByYear.keys()].sort((a, b) => a - b);

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
        {labelYears.map((y, idx) => {
          // 大间隔(超过 3 个空白年份)在轴上画一个虚线提示
          const prev = labelYears[idx - 1];
          const gap = prev != null && y - prev > 3;
          return (
            <g key={y}>
              {gap && (
                <line
                  x1={xScale(prev) + 6}
                  x2={xScale(y) - 6}
                  y1={AXIS_Y}
                  y2={AXIS_Y}
                  stroke="var(--ink-muted)"
                  strokeWidth={1}
                  strokeDasharray="3 4"
                  opacity={0.5}
                />
              )}
              <g transform={`translate(${xScale(y)}, ${AXIS_Y})`}>
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
            </g>
          );
        })}
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
              const scaleX = rect.width / width;
              const scaleY = rect.height / height;
              setPos({
                x: rect.left + window.scrollX + xScale(n.year) * scaleX - 120,
                y: rect.top + window.scrollY + yByNode.get(n.path)! * scaleY - 200,
              });
            }}
            onMouseLeave={() => setHovered(null)}
            onClick={() => {
              const slug = n.path.split("/").pop()!.replace(/\.md$/, "");
              navigate(`/families/${n.family}/${slug}`);
            }}
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
