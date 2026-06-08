import type { MiniArchProps } from "./types";

export function MiniConvNeXt({
  width = 160,
  height = 70,
  ariaLabel = "ConvNeXt 架构缩图（占位）",
}: MiniArchProps) {
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <rect
        x="10"
        y="20"
        width="140"
        height="30"
        rx="4"
        fill="#f4f4f5"
        stroke="#a1a1aa"
        strokeWidth="1"
      />
      <text
        x="80"
        y="40"
        textAnchor="middle"
        fontSize="11"
        fill="#71717a"
      >
        ConvNeXt · 占位
      </text>
    </svg>
  );
}
