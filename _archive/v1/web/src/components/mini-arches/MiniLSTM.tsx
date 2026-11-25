import type { MiniArchProps } from "./types";

export function MiniLSTM({
  width = 160,
  height = 70,
  ariaLabel = "LSTM 架构缩图",
}: MiniArchProps) {
  // 顶层:细胞状态高速公路(粗直线 + 三个门控圆作为乘法节点)
  // 底层:隐状态横向流(虚线)
  const gates = [40, 80, 120]; // f / i / o 三道门
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 细胞状态高速公路 — 顶部粗实线 */}
        <line
          x1="6"
          y1="20"
          x2="154"
          y2="20"
          className="illustration__branch illustration__branch--k"
          strokeDasharray="0"
        />
        {/* 三道门作为圆形乘法节点 */}
        {gates.map((cx, i) => (
          <circle
            key={i}
            cx={cx}
            cy="20"
            r="5"
            className="illustration__addnorm"
          />
        ))}
        {/* 底层隐状态主体 — 一个细长方块 */}
        <rect
          x="6"
          y="40"
          width="148"
          height="18"
          rx="3"
          className="illustration__proj illustration__proj--ffn"
        />
        {/* 三道门往下连到隐状态 — 控制信号 */}
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
