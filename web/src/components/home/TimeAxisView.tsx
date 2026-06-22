import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router";
import type { FamiliesData, FamilyId, NodeData } from "../../types/family";
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
  // 家族多选筛选:空集 = 显示全部;有选 = 只显示选中的家族
  const [selectedFamilies, setSelectedFamilies] = useState<Set<FamilyId>>(
    new Set()
  );
  const toggleFamily = (fid: FamilyId) => {
    setSelectedFamilies((prev) => {
      const next = new Set(prev);
      if (next.has(fid)) next.delete(fid);
      else next.add(fid);
      return next;
    });
  };
  const showAll = selectedFamilies.size === 0;
  const allNodes = useMemo(
    () =>
      data.families
        .filter((f) => showAll || selectedFamilies.has(f.id))
        .flatMap((f) => f.nodes),
    [data, showAll, selectedFamilies]
  );

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

  // 每个节点的标签可用横向空间 — 取到下一个有节点年份的距离,留出 padding;
  // 把名字截到能塞下的字符数(汉字 ≈ 10px,英数 ≈ 6.5px,这里按英数估)
  const labelMaxChars = (year: number): number => {
    const idx = labelYears.indexOf(year);
    const nextYear = idx >= 0 && idx < labelYears.length - 1 ? labelYears[idx + 1] : null;
    const thisX = xScale(year);
    const nextX = nextYear != null ? xScale(nextYear) : width - PADDING;
    const availPx = nextX - thisX - NODE_RADIUS - 6;
    return Math.max(3, Math.floor(availPx / 6.5));
  };
  const truncLabel = (name: string, year: number): string => {
    const max = labelMaxChars(year);
    if (name.length <= max) return name;
    return name.slice(0, Math.max(1, max - 1)) + "…";
  };

  return (
    <div style={{ position: "relative" }}>
      <div
        role="group"
        aria-label="按家族筛选"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "var(--space-2)",
          marginBottom: "var(--space-4)",
          justifyContent: "center",
        }}
      >
        {data.families.map((f) => {
          const active = selectedFamilies.has(f.id);
          const color = familyColorVar(f.id);
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => toggleFamily(f.id)}
              aria-pressed={active}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 10px",
                fontSize: "var(--fs-sm)",
                borderRadius: "var(--radius-full)",
                border: `1px solid ${active ? color : "var(--border)"}`,
                background: active ? color : "var(--bg-surface)",
                color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
                cursor: "pointer",
                transition: "all var(--dur-fast) var(--ease-out)",
              }}
            >
              <span
                aria-hidden="true"
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: active ? "var(--bg-surface)" : color,
                }}
              />
              {f.label}
            </button>
          );
        })}
        {selectedFamilies.size > 0 && (
          <button
            type="button"
            onClick={() => setSelectedFamilies(new Set())}
            style={{
              padding: "4px 10px",
              fontSize: "var(--fs-sm)",
              borderRadius: "var(--radius-full)",
              border: "1px dashed var(--ink-muted)",
              background: "transparent",
              color: "var(--ink-muted)",
              cursor: "pointer",
            }}
          >
            清除筛选
          </button>
        )}
      </div>
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
        {allNodes.map((n) => {
          const cx = xScale(n.year);
          const cy = yByNode.get(n.path)!;
          const goTo = () => {
            const slug = n.path.split("/").pop()!.replace(/\.md$/, "");
            navigate(`/families/${n.family}/${slug}`);
          };
          const positionCard = (target: SVGGElement) => {
            const svgEl = target.ownerSVGElement!;
            const rect = svgEl.getBoundingClientRect();
            const scaleX = rect.width / width;
            const scaleY = rect.height / height;
            setPos({
              x: rect.left + window.scrollX + cx * scaleX - 120,
              y: rect.top + window.scrollY + cy * scaleY - 200,
            });
          };
          return (
            <g
              key={n.path}
              role="link"
              tabIndex={0}
              aria-label={`${n.name} (${n.year}) — ${n.key_idea}`}
              style={{ cursor: "pointer", outline: "none" }}
              onMouseEnter={(e) => {
                setHovered(n);
                positionCard(e.currentTarget as SVGGElement);
              }}
              onMouseLeave={() => setHovered(null)}
              onFocus={(e) => {
                setHovered(n);
                positionCard(e.currentTarget as SVGGElement);
              }}
              onBlur={() => setHovered(null)}
              onClick={goTo}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  goTo();
                }
              }}
            >
              <motion.circle
                cx={cx}
                cy={cy}
                r={NODE_RADIUS}
                fill={familyColorVar(n.family)}
                stroke="var(--bg-canvas)"
                strokeWidth={2}
                whileHover={{ scale: 1.4 }}
                layoutId={`node-${n.path}`}
              />
              {/* 常驻标签 —— 节点名跟在圆点右侧,用家族色让密集区也能看出归属。
                  paint-order stroke→fill 给文字一圈 bg-canvas halo,
                  即使略和邻居重叠也保持可读;长度按到下一年的可用空间动态截 */}
              <text
                x={cx + NODE_RADIUS + 4}
                y={cy + 3}
                fontSize={10}
                fill={familyColorVar(n.family)}
                fontWeight={500}
                stroke="var(--bg-canvas)"
                strokeWidth={3}
                style={{
                  pointerEvents: "none",
                  userSelect: "none",
                  paintOrder: "stroke fill",
                }}
              >
                {truncLabel(n.name, n.year)}
              </text>
            </g>
          );
        })}
      </svg>
      <AnimatePresence>
        {hovered && <NodeHoverCard node={hovered} x={pos.x} y={pos.y} />}
      </AnimatePresence>
    </div>
  );
}
