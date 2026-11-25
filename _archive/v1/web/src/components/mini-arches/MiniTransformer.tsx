import type { MiniArchProps } from "./types";

export function MiniTransformer({
  width = 160,
  height = 70,
  ariaLabel = "Transformer 架构缩图",
}: MiniArchProps) {
  // 经典 encoder-decoder 双塔:左塔 encoder N 层、右塔 decoder N 层、
  // 中间一道 cross-attention 箭头
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Encoder 塔(左)— 三层堆叠 */}
        {[10, 22, 34].map((y, i) => (
          <rect
            key={`e-${i}`}
            x="14"
            y={y}
            width="40"
            height="10"
            rx="2"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* Decoder 塔(右)— 三层堆叠 */}
        {[10, 22, 34].map((y, i) => (
          <rect
            key={`d-${i}`}
            x="106"
            y={y}
            width="40"
            height="10"
            rx="2"
            className="illustration__proj illustration__proj--v"
          />
        ))}
        {/* Cross-attention 主箭头(encoder 顶层 → decoder 中层) */}
        <line
          x1="54"
          y1="15"
          x2="106"
          y2="27"
          className="illustration__branch illustration__branch--q"
        />
        {/* 输入/输出方块 */}
        <rect
          x="14"
          y="52"
          width="40"
          height="10"
          rx="2"
          className="illustration__layer illustration__layer--input"
        />
        <rect
          x="106"
          y="52"
          width="40"
          height="10"
          rx="2"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
      </g>
    </svg>
  );
}
