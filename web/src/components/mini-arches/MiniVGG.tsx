import type { MiniArchProps } from "./types";

export function MiniVGG({
  width = 160,
  height = 70,
  ariaLabel = "VGG 架构缩图",
}: MiniArchProps) {
  // 强调"深而整齐"：一长串等大方块
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {Array.from({ length: 13 }).map((_, i) => (
          <rect key={i} x={i * 9} y="22" width="7" height="32" rx="1.5"
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
