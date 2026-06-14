import type { MiniArchProps } from "./types";

export function MiniDPO({
  width = 160,
  height = 70,
  ariaLabel = "DPO 架构缩图",
}: MiniArchProps) {
  // (winner, loser) 偏好对 → 一个 LLM → 简单 loss
  // 突出"没有 RM、没有 critic、一个模型 + 监督损失"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:偏好对 — winner(绿)+ loser(灰/淡) */}
        <rect
          x="4"
          y="14"
          width="32"
          height="14"
          rx="2"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <text
          x="20"
          y="23"
          textAnchor="middle"
          fontSize="6"
          fontWeight="600"
          fill="currentColor"
          opacity="0.8"
        >
          y_w ✓
        </text>
        <rect
          x="4"
          y="42"
          width="32"
          height="14"
          rx="2"
          className="illustration__proj illustration__proj--ffn"
          opacity="0.45"
        />
        <text
          x="20"
          y="51"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.6"
        >
          y_l ✗
        </text>
        {/* 中央:单个 LLM(对比 RLHF 4 个模型,DPO 只要 actor + 冻结的 ref) */}
        <rect
          x="58"
          y="20"
          width="44"
          height="30"
          rx="3"
          className="illustration__proj illustration__proj--v"
        />
        <text
          x="80"
          y="33"
          textAnchor="middle"
          fontSize="7"
          fontWeight="600"
          fill="currentColor"
          opacity="0.85"
        >
          π_θ
        </text>
        <text
          x="80"
          y="44"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.7"
        >
          (+ ref)
        </text>
        {/* 右侧:简单 loss 公式块 */}
        <rect
          x="120"
          y="24"
          width="36"
          height="22"
          rx="2"
          fill="none"
          className="illustration__block"
        />
        <text
          x="138"
          y="34"
          textAnchor="middle"
          fontSize="6"
          fontWeight="600"
          fill="currentColor"
          opacity="0.8"
        >
          −log σ
        </text>
        <text
          x="138"
          y="42"
          textAnchor="middle"
          fontSize="5"
          fill="currentColor"
          opacity="0.65"
        >
          (β·Δ)
        </text>
        {/* 输入到 LLM 的箭头 */}
        <line
          x1="36"
          y1="21"
          x2="58"
          y2="28"
          className="illustration__branch illustration__branch--q"
        />
        <line
          x1="36"
          y1="49"
          x2="58"
          y2="42"
          className="illustration__branch illustration__branch--q"
        />
        {/* LLM 到 loss 的箭头 */}
        <line
          x1="102"
          y1="35"
          x2="120"
          y2="35"
          className="illustration__branch illustration__branch--v"
        />
        {/* 底部强调 "无 RM 无 PPO" */}
        <text
          x="80"
          y="64"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          无 RM · 无 PPO · 监督损失
        </text>
      </g>
    </svg>
  );
}
