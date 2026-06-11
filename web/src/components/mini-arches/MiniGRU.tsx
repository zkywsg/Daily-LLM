import type { MiniArchProps } from "./types";

export function MiniGRU({
  width = 160,
  height = 70,
  ariaLabel = "GRU 架构缩图",
}: MiniArchProps) {
  // 和 LSTM 同款拓扑但只有两道门 — 视觉上"更简"
  const gates = [55, 105]; // r / z 两道门
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 单一隐状态 highway */}
        <line
          x1="6"
          y1="20"
          x2="154"
          y2="20"
          className="illustration__branch illustration__branch--k"
          strokeDasharray="0"
        />
        {gates.map((cx, i) => (
          <circle
            key={i}
            cx={cx}
            cy="20"
            r="5"
            className="illustration__addnorm"
          />
        ))}
        {/* 隐状态主体 */}
        <rect
          x="6"
          y="40"
          width="148"
          height="18"
          rx="3"
          className="illustration__proj illustration__proj--ffn"
        />
        {gates.map((cx, i) => (
          <line
            key={`g-${i}`}
            x1={cx}
            y1="25"
            x2={cx}
            y2="40"
            className="illustration__branch illustration__branch--v"
          />
        ))}
      </g>
    </svg>
  );
}
