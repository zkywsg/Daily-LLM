import type { MiniArchProps } from "./types";

export function MiniEfficientNet({
  width = 160,
  height = 70,
  ariaLabel = "EfficientNet 架构缩图",
}: MiniArchProps) {
  // 复合缩放：MBConv 块沿 depth/width 同步放大 + 上扬的缩放弧线
  const blocks = [
    { x: 6, w: 14, h: 12 },
    { x: 26, w: 17, h: 16 },
    { x: 49, w: 20, h: 21 },
    { x: 75, w: 23, h: 27 },
    { x: 104, w: 26, h: 34 },
  ];
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {blocks.map((b, i) => (
          <rect key={i} x={b.x} y={56 - b.h} width={b.w} height={b.h} rx="3"
            className="illustration__proj illustration__proj--ffn" />
        ))}
        {/* 复合缩放轨迹 */}
        <path d="M 10 38 C 40 30, 95 22, 134 12"
          className="illustration__residual" fill="none" />
        <path d="M 134 12 l -8 -1 M 134 12 l -3 7"
          className="illustration__residual" fill="none" />
      </g>
    </svg>
  );
}
