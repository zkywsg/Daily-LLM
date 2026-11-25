import { useState } from "react";

interface Props {
  width?: number;
  height?: number;
}

const BASE_CHANNELS = 256; // Dense Block 3 输入通道(DenseNet-121)
const GROWTH_K = 32;
const NUM_LAYERS = 24; // Dense Block 3 层数

/** 第 ℓ 层(1-indexed)的累积输入通道数 = base + (ℓ-1)*k */
function channelsAt(layer: number): number {
  return BASE_CHANNELS + (layer - 1) * GROWTH_K;
}

/**
 * 折线图:Dense Block 3(DenseNet-121,24 层,k=32)通道数随层号线性增长,
 * 最终从 256 涨到 992 —— 对应正文 "256 + 23×32 = 992" 的具体数字。
 * 悬停任意点显示该层的累积输入通道数。
 */
export function ChannelGrowthChart({ width = 560, height = 320 }: Props) {
  const padding = { top: 30, right: 30, bottom: 50, left: 60 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const maxY = channelsAt(NUM_LAYERS);
  const layers = Array.from({ length: NUM_LAYERS }, (_, i) => i + 1);

  const xFor = (layer: number) => padding.left + (plotW * (layer - 1)) / (NUM_LAYERS - 1);
  const yFor = (val: number) => padding.top + plotH - (plotH * val) / maxY;

  const pathD = layers
    .map((l, i) => `${i === 0 ? "M" : "L"} ${xFor(l)} ${yFor(channelsAt(l))}`)
    .join(" ");

  const [hoverLayer, setHoverLayer] = useState<number | null>(null);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="Dense Block 3 通道数随层号线性增长折线图"
    >
      <text x={width / 2} y={18} textAnchor="middle" fontSize={14} fontWeight={600} fill="var(--ink-primary)">
        Dense Block 3(24 层,k=32)—— 通道数线性增长,不爆炸
      </text>

      {/* Y 轴 */}
      <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + plotH} stroke="#9ca3af" strokeWidth={1} />
      <text x={padding.left - 8} y={padding.top + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">{maxY}</text>
      <text x={padding.left - 8} y={padding.top + plotH} textAnchor="end" fontSize={10} fill="var(--ink-muted)">0</text>

      {/* X 轴 */}
      <line x1={padding.left} y1={padding.top + plotH} x2={padding.left + plotW} y2={padding.top + plotH} stroke="#9ca3af" strokeWidth={1} />
      <text x={padding.left} y={padding.top + plotH + 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">层 1</text>
      <text x={padding.left + plotW} y={padding.top + plotH + 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">层 24</text>

      {/* 折线 */}
      <path d={pathD} fill="none" stroke="#3b82f6" strokeWidth={2.5} />

      {/* 数据点 */}
      {layers.map((l) => (
        <circle
          key={l}
          cx={xFor(l)}
          cy={yFor(channelsAt(l))}
          r={hoverLayer === l ? 5 : 3}
          fill={hoverLayer === l ? "#1d4ed8" : "#3b82f6"}
          onMouseEnter={() => setHoverLayer(l)}
          onMouseLeave={() => setHoverLayer(null)}
          style={{ cursor: "help" }}
        />
      ))}

      {/* 起点/终点标注 */}
      <text x={xFor(1)} y={yFor(channelsAt(1)) - 12} textAnchor="middle" fontSize={11} fill="#1d4ed8">
        {channelsAt(1)}ch
      </text>
      <text x={xFor(NUM_LAYERS)} y={yFor(channelsAt(NUM_LAYERS)) - 12} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1d4ed8">
        {channelsAt(NUM_LAYERS)}ch
      </text>

      {/* 悬停提示 */}
      {hoverLayer !== null && (
        <g transform={`translate(${Math.min(Math.max(xFor(hoverLayer), 70), width - 70)}, ${padding.top + plotH + 38})`}>
          <text textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--ink-primary)">
            第 {hoverLayer} 层输入 = {BASE_CHANNELS} + {hoverLayer - 1}×{GROWTH_K} = {channelsAt(hoverLayer)}ch
          </text>
        </g>
      )}
    </svg>
  );
}
