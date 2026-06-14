import type { MiniArchProps } from "./types";

export function MiniLearningToSummarize({
  width = 160,
  height = 70,
  ariaLabel = "Learning to Summarize 架构缩图",
}: MiniArchProps) {
  // 三阶段 pipeline: SFT → RM → PPO,横向三个 box
  const stages = [
    { x: 4, label: "SFT", cls: "illustration__layer illustration__layer--input" },
    { x: 60, label: "RM", cls: "illustration__proj illustration__proj--ffn" },
    { x: 116, label: "PPO", cls: "illustration__featuremap illustration__featuremap--ctx" },
  ];
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {stages.map((s, i) => (
          <g key={i}>
            <rect
              x={s.x}
              y="22"
              width="40"
              height="26"
              rx="3"
              className={s.cls}
            />
            <text
              x={s.x + 20}
              y="40"
              textAnchor="middle"
              fontSize="9"
              fontWeight="600"
              fill="currentColor"
              opacity="0.8"
            >
              {s.label}
            </text>
          </g>
        ))}
        {/* 连接箭头 */}
        <line
          x1="44"
          y1="35"
          x2="60"
          y2="35"
          className="illustration__branch illustration__branch--q"
        />
        <line
          x1="100"
          y1="35"
          x2="116"
          y2="35"
          className="illustration__branch illustration__branch--q"
        />
        {/* 顶部 label */}
        <text
          x="24"
          y="14"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          阶段 1
        </text>
        <text
          x="80"
          y="14"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          阶段 2
        </text>
        <text
          x="136"
          y="14"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          阶段 3
        </text>
        {/* 底部输出 */}
        <text
          x="80"
          y="62"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          人工偏好 → 模型对齐
        </text>
      </g>
    </svg>
  );
}
