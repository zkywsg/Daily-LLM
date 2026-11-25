import type { MiniArchProps } from "./types";

export function MiniConstitutionalAI({
  width = 160,
  height = 70,
  ariaLabel = "Constitutional AI 架构缩图",
}: MiniArchProps) {
  // 一个 LLM 同时扮演"作答"和"自评"两个角色 — 体现 AI feedback
  // 左侧:LLM 生成回答;中央:constitution(原则列表);右侧:LLM 自评后的修正回答
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧 LLM(初始回答) */}
        <rect
          x="4"
          y="20"
          width="32"
          height="30"
          rx="3"
          className="illustration__proj illustration__proj--ffn"
        />
        <text
          x="20"
          y="38"
          textAnchor="middle"
          fontSize="7"
          fontWeight="600"
          fill="currentColor"
          opacity="0.8"
        >
          LLM
        </text>
        <text
          x="20"
          y="60"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          回答
        </text>
        {/* 中央 constitution(原则列表) */}
        <rect
          x="56"
          y="14"
          width="48"
          height="42"
          rx="3"
          fill="none"
          className="illustration__block"
          strokeDasharray="2,1"
        />
        {/* 几行模拟原则文字 */}
        {[22, 28, 34, 40, 46].map((y, i) => (
          <line
            key={i}
            x1="62"
            y1={y}
            x2={94 - i * 2}
            y2={y}
            stroke="currentColor"
            strokeWidth="0.8"
            opacity="0.45"
          />
        ))}
        <text
          x="80"
          y="64"
          textAnchor="middle"
          fontSize="6"
          fontWeight="600"
          fill="currentColor"
          opacity="0.7"
        >
          constitution
        </text>
        {/* 右侧 LLM(自评 + 修正) */}
        <rect
          x="124"
          y="20"
          width="32"
          height="30"
          rx="3"
          className="illustration__proj illustration__proj--v"
        />
        <text
          x="140"
          y="35"
          textAnchor="middle"
          fontSize="7"
          fontWeight="600"
          fill="currentColor"
          opacity="0.85"
        >
          LLM
        </text>
        <text
          x="140"
          y="44"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.7"
        >
          自评
        </text>
        {/* 箭头:LLM → constitution → 自评 → 回退到 LLM(critique-revise 循环) */}
        <line
          x1="36"
          y1="35"
          x2="56"
          y2="35"
          className="illustration__branch illustration__branch--q"
        />
        <line
          x1="104"
          y1="35"
          x2="124"
          y2="35"
          className="illustration__branch illustration__branch--v"
        />
        {/* 回环:右侧 LLM → 左侧 LLM(虚线,体现自我迭代) */}
        <path
          d="M 140 50 Q 80 68, 20 50"
          fill="none"
          className="illustration__residual"
        />
      </g>
    </svg>
  );
}
