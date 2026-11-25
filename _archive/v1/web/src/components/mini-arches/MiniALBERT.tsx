import type { MiniArchProps } from "./types";

export function MiniALBERT({
  width = 160,
  height = 70,
  ariaLabel = "ALBERT 架构缩图",
}: MiniArchProps) {
  // 跨层参数共享:一个 layer 实例 + 循环弧线表示重复用 N 次
  // 视觉:中央一个粗块标"shared layer",外面虚线弧 ×N
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 输入(左) */}
        <rect
          x="4"
          y="30"
          width="20"
          height="10"
          rx="1"
          className="illustration__layer illustration__layer--input"
        />
        {/* 输出(右) */}
        <rect
          x="136"
          y="30"
          width="20"
          height="10"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        {/* 中央"共享 layer"块,更粗 */}
        <rect
          x="58"
          y="22"
          width="44"
          height="26"
          rx="3"
          className="illustration__proj illustration__proj--v"
        />
        <text
          x="80"
          y="34"
          textAnchor="middle"
          fontSize="7"
          fontWeight="600"
          fill="currentColor"
          opacity="0.85"
        >
          shared
        </text>
        <text
          x="80"
          y="42"
          textAnchor="middle"
          fontSize="7"
          fontWeight="600"
          fill="currentColor"
          opacity="0.85"
        >
          layer
        </text>
        {/* 循环弧线:从右侧 → 顶/底 → 回到左侧,表示"forward 多次" */}
        <path
          d="M 102 28 C 130 6, 130 6, 102 22"
          fill="none"
          className="illustration__residual"
        />
        <text
          x="120"
          y="14"
          textAnchor="middle"
          fontSize="7"
          fontWeight="600"
          fill="currentColor"
          opacity="0.7"
        >
          ×N
        </text>
        {/* 输入到 layer 的箭头 */}
        <line
          x1="24"
          y1="35"
          x2="58"
          y2="35"
          className="illustration__branch illustration__branch--q"
        />
        {/* layer 到输出 */}
        <line
          x1="102"
          y1="48"
          x2="136"
          y2="40"
          className="illustration__branch illustration__branch--v"
        />
        {/* 底部:embedding 因式分解的视觉小提示 */}
        <text
          x="80"
          y="64"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          参数共享 · V×E + E×H
        </text>
      </g>
    </svg>
  );
}
