import type { MiniArchProps } from "./types";

export function MiniGPT1({
  width = 160,
  height = 70,
  ariaLabel = "GPT-1 架构缩图",
}: MiniArchProps) {
  // Decoder-only 12 层 stack(竖向堆叠)+ 左边输入 + 右边输出
  // 体现"单塔自回归"特征(对比 Transformer 双塔)
  const layers = 12;
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 中央 decoder stack */}
        {Array.from({ length: layers }, (_, i) => (
          <rect
            key={i}
            x="62"
            y={8 + i * 4.2}
            width="36"
            height="3"
            rx="1"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* 输入(下,琥珀) */}
        <rect
          x="62"
          y="60"
          width="36"
          height="6"
          rx="2"
          className="illustration__layer illustration__layer--input"
        />
        {/* 输出(上,绿) */}
        <rect
          x="62"
          y="2"
          width="36"
          height="4"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        {/* 左侧 input prompt 排 — 体现 token 序列 */}
        {[24, 32, 40, 48].map((y, i) => (
          <rect
            key={i}
            x="6"
            y={y}
            width="14"
            height="6"
            rx="1"
            className="illustration__layer illustration__layer--input"
            opacity={0.8 - i * 0.15}
          />
        ))}
        <line
          x1="22"
          y1="38"
          x2="62"
          y2="38"
          className="illustration__branch illustration__branch--q"
        />
        {/* 右侧 output token — 自回归生成 */}
        <rect
          x="138"
          y="35"
          width="14"
          height="6"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <line
          x1="100"
          y1="38"
          x2="138"
          y2="38"
          className="illustration__branch illustration__branch--v"
        />
      </g>
    </svg>
  );
}
