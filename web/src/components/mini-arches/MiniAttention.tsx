import type { MiniArchProps } from "./types";

export function MiniAttention({
  width = 160,
  height = 70,
  ariaLabel = "Bahdanau Attention 架构缩图",
}: MiniArchProps) {
  // 底排 encoder 全部时刻 + 顶部 decoder 当前步 + 从 decoder 发散到 encoder 的多条加权线
  const enc = [10, 34, 58, 82, 106, 130]; // 6 个 encoder 时刻
  const decX = 134;
  const decY = 14;
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 底排 encoder 时刻 */}
        {enc.map((x, i) => (
          <rect
            key={`e-${i}`}
            x={x}
            y="48"
            width="14"
            height="12"
            rx="2"
            className="illustration__layer illustration__layer--input"
          />
        ))}
        {/* 顶部 decoder 当前步 */}
        <rect
          x={decX}
          y={decY}
          width="16"
          height="14"
          rx="2"
          className="illustration__proj illustration__proj--act"
        />
        {/* 从 decoder 发散到所有 encoder 时刻 — 注意力权重 */}
        {enc.map((x, i) => {
          // 中间几个权重大(用实线),两端权重小(用虚线表示淡化)
          const isStrong = i === 2 || i === 3;
          return (
            <line
              key={`a-${i}`}
              x1={decX + 8}
              y1={decY + 14}
              x2={x + 7}
              y2={48}
              className={
                isStrong
                  ? "illustration__branch illustration__branch--v"
                  : "illustration__residual"
              }
              fill="none"
            />
          );
        })}
      </g>
    </svg>
  );
}
