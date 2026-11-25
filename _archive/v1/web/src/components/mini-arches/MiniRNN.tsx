import type { MiniArchProps } from "./types";

export function MiniRNN({
  width = 160,
  height = 70,
  ariaLabel = "RNN 架构缩图",
}: MiniArchProps) {
  // 4 个隐状态方块横向连接;每个方块上方一条循环回路弧表示自指
  const cells = [
    { x: 12, label: "h₁" },
    { x: 50, label: "h₂" },
    { x: 88, label: "h₃" },
    { x: 126, label: "h₄" },
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
        {/* 隐状态方块 */}
        {cells.map((c, i) => (
          <rect
            key={i}
            x={c.x}
            y="30"
            width="22"
            height="18"
            rx="2"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* 横向连接箭头 h_{t-1} → h_t */}
        {[0, 1, 2].map((i) => (
          <line
            key={i}
            x1={cells[i].x + 22}
            y1="39"
            x2={cells[i + 1].x}
            y2="39"
            className="illustration__branch illustration__branch--q"
          />
        ))}
        {/* 每个方块顶上一条自指小弧 — 循环的视觉签名 */}
        {cells.map((c, i) => (
          <path
            key={`loop-${i}`}
            d={`M ${c.x + 6} 30 C ${c.x + 6} 16, ${c.x + 16} 16, ${c.x + 16} 30`}
            className="illustration__residual"
            fill="none"
          />
        ))}
      </g>
    </svg>
  );
}
