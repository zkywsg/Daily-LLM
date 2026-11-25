import type { MiniArchProps } from "./types";

export function MiniGPT4Llama({
  width = 160,
  height = 70,
  ariaLabel = "GPT-4 / LLaMA 架构缩图",
}: MiniArchProps) {
  // MoE 视觉:中央 router → 8 个 expert 块(典型 GPT-4 配置 16x110B,
  // 这里简化成 8 个),其中 2 个被选中(高亮)体现 sparse activation
  const experts = [
    { x: 60, y: 4, active: false },
    { x: 80, y: 4, active: true },
    { x: 100, y: 4, active: false },
    { x: 120, y: 4, active: false },
    { x: 60, y: 22, active: false },
    { x: 80, y: 22, active: false },
    { x: 100, y: 22, active: true },
    { x: 120, y: 22, active: false },
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
        {/* 左侧输入 */}
        <rect
          x="4"
          y="46"
          width="14"
          height="6"
          rx="1"
          className="illustration__layer illustration__layer--input"
        />
        {/* Router(中下) */}
        <circle
          cx="38"
          cy="49"
          r="6"
          className="illustration__addnorm"
        />
        <text
          x="38"
          y="51"
          textAnchor="middle"
          fontSize="5"
          fill="currentColor"
          opacity="0.8"
        >
          R
        </text>
        <line
          x1="18"
          y1="49"
          x2="32"
          y2="49"
          className="illustration__branch illustration__branch--q"
        />
        {/* Experts grid 2×4 — 部分高亮表示稀疏激活 */}
        {experts.map((e, i) => (
          <rect
            key={i}
            x={e.x}
            y={e.y}
            width="14"
            height="14"
            rx="2"
            className={
              e.active
                ? "illustration__proj illustration__proj--v"
                : "illustration__proj illustration__proj--ffn"
            }
            opacity={e.active ? 1 : 0.35}
          />
        ))}
        {/* Router → 激活的 experts 的虚线 */}
        {experts
          .filter((e) => e.active)
          .map((e, i) => (
            <line
              key={`act-${i}`}
              x1="44"
              y1="49"
              x2={e.x + 7}
              y2={e.y + 14}
              className="illustration__branch illustration__branch--v"
            />
          ))}
        {/* 右下输出 */}
        <rect
          x="142"
          y="46"
          width="14"
          height="6"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <line
          x1="135"
          y1="49"
          x2="142"
          y2="49"
          className="illustration__branch illustration__branch--k"
        />
      </g>
    </svg>
  );
}
