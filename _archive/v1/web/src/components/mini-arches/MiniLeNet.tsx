import type { MiniArchProps } from "./types";

export function MiniLeNet({
  width = 160,
  height = 70,
  ariaLabel = "LeNet 架构缩图",
}: MiniArchProps) {
  // conv→pool 交替收缩的特征图 + 末端全连接细条
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        <rect x="4" y="21" width="28" height="28" rx="2"
          className="illustration__layer illustration__layer--input" />
        {[
          { x: 42, s: 22 },
          { x: 72, s: 16 },
          { x: 94, s: 11 },
        ].map((b, i) => (
          <rect key={i} x={b.x} y={35 - b.s / 2} width={b.s} height={b.s} rx="2"
            className="illustration__layer illustration__layer--conv" />
        ))}
        {[118, 130, 142].map((x, i) => (
          <rect key={x} x={x} y={26 + i * 2} width="5" height={18 - i * 4} rx="1"
            className="illustration__layer illustration__layer--fc" />
        ))}
      </g>
    </svg>
  );
}
