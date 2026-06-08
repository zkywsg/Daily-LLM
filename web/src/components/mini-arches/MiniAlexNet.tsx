import type { MiniArchProps } from "./types";

export function MiniAlexNet({
  width = 160,
  height = 70,
  ariaLabel = "AlexNet 架构缩图",
}: MiniArchProps) {
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 5 conv 块（递减）+ 3 fc 细条 */}
        {[
          { x: 0, w: 28 }, { x: 32, w: 22 }, { x: 58, w: 18 },
          { x: 80, w: 16 }, { x: 100, w: 14 },
        ].map((b, i) => (
          <rect key={i} x={b.x} y={26 - b.w / 2 + 14} width={b.w} height={b.w} rx="2"
            className="illustration__layer illustration__layer--conv" />
        ))}
        {[120, 130, 140].map((x) => (
          <rect key={x} x={x} y="22" width="6" height="32" rx="1"
            className="illustration__layer illustration__layer--fc" />
        ))}
      </g>
    </svg>
  );
}
