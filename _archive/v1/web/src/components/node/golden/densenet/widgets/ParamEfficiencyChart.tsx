import { useState } from "react";
import { MODEL_COMPARISON, type ModelComparisonPoint } from "../lib/data";

interface Props {
  width?: number;
  height?: number;
}

/**
 * 散点图:参数量(M)vs Top-5 错误率(%),ResNet 系列 vs DenseNet 系列。
 * DenseNet 的点整体位于 ResNet 左下方 —— 更少参数达到更低错误率。
 */
export function ParamEfficiencyChart({ width = 560, height = 360 }: Props) {
  const padding = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const maxParams = 65;
  const minError = 4.5;
  const maxError = 7.2;

  const xFor = (p: number) => padding.left + (plotW * p) / maxParams;
  const yFor = (e: number) =>
    padding.top + plotH - (plotH * (e - minError)) / (maxError - minError);

  const [hover, setHover] = useState<ModelComparisonPoint | null>(null);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="DenseNet 与 ResNet 参数量 vs Top-5 错误率对比散点图"
    >
      <text x={width / 2} y={18} textAnchor="middle" fontSize={14} fontWeight={600} fill="var(--ink-primary)">
        参数量 vs Top-5 错误率(ImageNet)
      </text>

      {/* 坐标轴 */}
      <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotH} stroke="#9ca3af" strokeWidth={1} />
      <line x1={padding.left} y1={padding.top + plotH} x2={padding.left + plotW} y2={padding.top + plotH} stroke="#9ca3af" strokeWidth={1} />

      <text x={padding.left - 10} y={yFor(maxError) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">{maxError}%</text>
      <text x={padding.left - 10} y={yFor(minError) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">{minError}%</text>
      <text x={padding.left} y={padding.top + plotH + 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">0M</text>
      <text x={padding.left + plotW} y={padding.top + plotH + 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">{maxParams}M</text>

      <text
        x={-(padding.top + plotH / 2)}
        y={16}
        transform="rotate(-90)"
        textAnchor="middle"
        fontSize={11}
        fill="var(--ink-secondary)"
      >
        Top-5 错误率
      </text>
      <text x={padding.left + plotW / 2} y={height - 8} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
        参数量(百万)
      </text>

      {/* 数据点 */}
      {MODEL_COMPARISON.map((pt, i) => {
        const isDense = pt.family === "densenet";
        const fill = isDense ? "#ec4899" : "#3b82f6";
        const isHover = hover?.model === pt.model;
        return (
          <g key={`${pt.model}-${i}`}>
            <circle
              cx={xFor(pt.paramsM)}
              cy={yFor(pt.top5Error)}
              r={isHover ? 9 : 7}
              fill={fill}
              opacity={0.85}
              stroke={isHover ? "var(--ink-primary)" : "none"}
              strokeWidth={2}
              onMouseEnter={() => setHover(pt)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: "help" }}
            />
            <text
              x={xFor(pt.paramsM)}
              y={yFor(pt.top5Error) - 12}
              textAnchor="middle"
              fontSize={9}
              fill="var(--ink-secondary)"
            >
              {pt.model}
            </text>
          </g>
        );
      })}

      {/* 图例 */}
      <g transform={`translate(${padding.left + 10}, ${padding.top + 4})`}>
        <circle cx={0} cy={0} r={5} fill="#3b82f6" />
        <text x={10} y={4} fontSize={10} fill="var(--ink-secondary)">ResNet</text>
        <circle cx={70} cy={0} r={5} fill="#ec4899" />
        <text x={80} y={4} fontSize={10} fill="var(--ink-secondary)">DenseNet</text>
      </g>

      {hover && (
        <g transform={`translate(${Math.min(Math.max(xFor(hover.paramsM), 80), width - 80)}, ${padding.top + plotH + 34})`}>
          <text textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--ink-primary)">
            {hover.model}: {hover.paramsM}M · {hover.top5Error}%
          </text>
        </g>
      )}
    </svg>
  );
}
