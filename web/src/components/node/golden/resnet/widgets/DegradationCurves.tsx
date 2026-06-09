import { useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { plainCurve, resnetCurve } from "../lib/curves";

interface Props {
  depth: number;
  width?: number;
  height?: number;
}

const PADDING = { top: 30, right: 30, bottom: 50, left: 60 };

interface HoverPoint {
  series: "plain" | "resnet";
  epoch: number;
  loss: number;
  px: number;
  py: number;
}

export function DegradationCurves({ depth, width = 560, height = 360 }: Props) {
  const plain = plainCurve(depth);
  const resnet = resnetCurve(depth);

  const innerW = width - PADDING.left - PADDING.right;
  const innerH = height - PADDING.top - PADDING.bottom;

  const xScale = scaleLinear().domain([0, 200]).range([0, innerW]);
  const yScale = scaleLinear().domain([0, 0.7]).range([innerH, 0]);

  const lineGen = line<{ epoch: number; loss: number }>()
    .x((d) => xScale(d.epoch))
    .y((d) => yScale(d.loss));

  const plainPath = lineGen(plain) ?? "";
  const resnetPath = lineGen(resnet) ?? "";

  const xTicks = [0, 50, 100, 150, 200];
  const yTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

  // Hover state
  const [hover, setHover] = useState<HoverPoint | null>(null);

  // Sample points along each curve for hover targets (every 10 epochs)
  const hoverTargets: HoverPoint[] = [];
  for (let i = 10; i < plain.length; i += 10) {
    const p = plain[i];
    hoverTargets.push({
      series: "plain",
      epoch: p.epoch,
      loss: p.loss,
      px: xScale(p.epoch),
      py: yScale(p.loss),
    });
  }
  for (let i = 10; i < resnet.length; i += 10) {
    const p = resnet[i];
    hoverTargets.push({
      series: "resnet",
      epoch: p.epoch,
      loss: p.loss,
      px: xScale(p.epoch),
      py: yScale(p.loss),
    });
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label={`训练损失曲线对比 ${depth} 层 plain 与 ResNet`}
    >
      <g transform={`translate(${PADDING.left}, ${PADDING.top})`}>
        <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="var(--border)" />
        <line x1={0} y1={0} x2={0} y2={innerH} stroke="var(--border)" />

        {xTicks.map((t) => (
          <g key={t} transform={`translate(${xScale(t)}, ${innerH})`}>
            <line y2={6} stroke="var(--ink-muted)" />
            <text y={20} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
              {t}
            </text>
          </g>
        ))}
        <text
          x={innerW / 2}
          y={innerH + 40}
          textAnchor="middle"
          fontSize={12}
          fill="var(--ink-secondary)"
        >
          epoch
        </text>

        {yTicks.map((t) => (
          <g key={t} transform={`translate(0, ${yScale(t)})`}>
            <line x2={-6} stroke="var(--ink-muted)" />
            <text x={-10} y={4} textAnchor="end" fontSize={11} fill="var(--ink-muted)">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          transform={`translate(-44, ${innerH / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={12}
          fill="var(--ink-secondary)"
        >
          training error
        </text>

        {/* 曲线 */}
        <path
          d={plainPath}
          stroke="#dc2626"
          strokeWidth={2}
          fill="none"
          strokeDasharray={depth > 30 ? "5,3" : "none"}
        />
        <path d={resnetPath} stroke="#2563eb" strokeWidth={2.5} fill="none" />

        {/* Hover 目标：透明 circle 覆盖在曲线上每隔 10 epoch 一个 */}
        {hoverTargets.map((p, i) => (
          <circle
            key={`hover-${p.series}-${i}`}
            cx={p.px}
            cy={p.py}
            r={10}
            fill="transparent"
            onMouseEnter={() => setHover(p)}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: "crosshair" }}
          />
        ))}

        {/* Hover 高亮：绘制 active 点的真实 dot */}
        {hover && (
          <>
            <circle
              cx={hover.px}
              cy={hover.py}
              r={5}
              fill={hover.series === "plain" ? "#dc2626" : "#2563eb"}
              stroke="var(--bg-canvas)"
              strokeWidth={2}
            />
            {/* Tooltip 背景 + 文字 */}
            <g
              transform={`translate(${Math.min(hover.px + 12, innerW - 130)}, ${Math.max(hover.py - 28, 0)})`}
              pointerEvents="none"
            >
              <rect
                x={0}
                y={0}
                width={130}
                height={48}
                rx={4}
                fill="var(--bg-surface)"
                stroke="var(--border)"
                strokeWidth={1}
              />
              <text x={8} y={18} fontSize={11} fill="var(--ink-secondary)">
                {hover.series === "plain" ? "plain" : "ResNet"}
              </text>
              <text x={8} y={34} fontSize={12} fill="var(--ink-primary)">
                epoch {hover.epoch} · {hover.loss.toFixed(3)}
              </text>
            </g>
          </>
        )}

        <g transform={`translate(${innerW - 140}, 10)`}>
          <line x1={0} y1={6} x2={20} y2={6} stroke="#dc2626" strokeWidth={2} />
          <text x={26} y={10} fontSize={12} fill="var(--ink-primary)">
            plain ({depth} 层)
          </text>
          <line x1={0} y1={26} x2={20} y2={26} stroke="#2563eb" strokeWidth={2.5} />
          <text x={26} y={30} fontSize={12} fill="var(--ink-primary)">
            ResNet ({depth} 层)
          </text>
        </g>
      </g>
    </svg>
  );
}
