import type { MiniArchProps } from "./types";

export function MiniScalingLaws({
  width = 160,
  height = 70,
  ariaLabel = "Scaling Laws 架构缩图",
}: MiniArchProps) {
  // log-log 坐标系上的 3 条幂律曲线 — L(N), L(D), L(C) 都按 N^-α 下降
  // 视觉上是 3 条向下倾斜的直线(log-log 下幂律是直线)
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 坐标轴 */}
        <line
          x1="12"
          y1="8"
          x2="12"
          y2="60"
          stroke="currentColor"
          strokeWidth="0.6"
          opacity="0.4"
        />
        <line
          x1="12"
          y1="60"
          x2="154"
          y2="60"
          stroke="currentColor"
          strokeWidth="0.6"
          opacity="0.4"
        />
        {/* Y 轴标签 */}
        <text
          x="4"
          y="14"
          fontSize="5"
          fill="currentColor"
          opacity="0.6"
        >
          loss
        </text>
        {/* X 轴标签 */}
        <text
          x="140"
          y="68"
          fontSize="5"
          fill="currentColor"
          opacity="0.6"
        >
          log scale
        </text>
        {/* 3 条幂律下降直线(不同斜率代表 N/D/C 三轴) */}
        <line
          x1="18"
          y1="14"
          x2="148"
          y2="44"
          className="illustration__branch illustration__branch--q"
        />
        <line
          x1="18"
          y1="22"
          x2="148"
          y2="48"
          className="illustration__branch illustration__branch--v"
        />
        <line
          x1="18"
          y1="30"
          x2="148"
          y2="52"
          className="illustration__branch illustration__branch--k"
        />
        {/* 三个标签 */}
        <text
          x="151"
          y="46"
          fontSize="5"
          fill="currentColor"
          opacity="0.7"
        >
          N
        </text>
        <text
          x="151"
          y="50"
          fontSize="5"
          fill="currentColor"
          opacity="0.7"
        >
          D
        </text>
        <text
          x="151"
          y="54"
          fontSize="5"
          fill="currentColor"
          opacity="0.7"
        >
          C
        </text>
      </g>
    </svg>
  );
}
