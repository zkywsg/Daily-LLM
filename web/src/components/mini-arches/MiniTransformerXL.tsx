import type { MiniArchProps } from "./types";

export function MiniTransformerXL({
  width = 160,
  height = 70,
  ariaLabel = "Transformer-XL 架构缩图",
}: MiniArchProps) {
  // 三段 Transformer 横向接力,每段间有 segment cache 虚线 — 体现段级循环
  const segs = [10, 60, 110]; // 三段 x 起点
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {segs.map((x, i) => (
          <g key={i}>
            {/* 每段 3 层 Transformer block */}
            {[16, 28, 40].map((y, j) => (
              <rect
                key={j}
                x={x}
                y={y}
                width="40"
                height="8"
                rx="2"
                className="illustration__proj illustration__proj--ffn"
              />
            ))}
            {/* 段标签下的输入条 */}
            <rect
              x={x}
              y="54"
              width="40"
              height="6"
              rx="1"
              className="illustration__layer illustration__layer--input"
            />
          </g>
        ))}
        {/* 段间 cache 虚线 — 段 i 的隐状态传到段 i+1 */}
        {[0, 1].map((i) => (
          <path
            key={`c-${i}`}
            d={`M ${segs[i] + 40} 32 Q ${(segs[i] + segs[i + 1] + 40) / 2} 8, ${segs[i + 1]} 32`}
            className="illustration__residual"
            fill="none"
          />
        ))}
      </g>
    </svg>
  );
}
