import type { MiniArchProps } from "./types";

export function MiniRoBERTa({
  width = 160,
  height = 70,
  ariaLabel = "RoBERTa 架构缩图",
}: MiniArchProps) {
  // BERT 形状基本相同,但 stack 更高(代表更长训练 + 更多数据)
  // 左侧多个数据来源箭头汇聚进 stack;无 NSP(没有第二段输入)
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Encoder stack 比 BERT 看起来更"密"(代表更长训练) */}
        {Array.from({ length: 16 }, (_, i) => (
          <rect
            key={i}
            x="60"
            y={5 + i * 3.6}
            width="48"
            height="2.4"
            rx="0.5"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* 左侧多个数据来源 — 体现 160GB 4 个数据集 */}
        {[
          { y: 14, label: "Book" },
          { y: 24, label: "CC-N" },
          { y: 34, label: "OWT" },
          { y: 44, label: "Wiki" },
        ].map((d, i) => (
          <g key={i}>
            <rect
              x="4"
              y={d.y}
              width="20"
              height="6"
              rx="1"
              className="illustration__layer illustration__layer--input"
            />
            <text
              x="14"
              y={d.y + 4.5}
              textAnchor="middle"
              fontSize="4.5"
              fill="currentColor"
              opacity="0.7"
            >
              {d.label}
            </text>
            {/* 数据汇聚箭头 */}
            <line
              x1="24"
              y1={d.y + 3}
              x2="60"
              y2="35"
              className="illustration__branch illustration__branch--q"
            />
          </g>
        ))}
        {/* 右侧输出 */}
        <rect
          x="138"
          y="30"
          width="18"
          height="10"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <line
          x1="108"
          y1="35"
          x2="138"
          y2="35"
          className="illustration__branch illustration__branch--v"
        />
        {/* 顶部标"4× 训练" */}
        <text
          x="84"
          y="66"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.55"
        >
          + 4× 训练 · 8K batch · 无 NSP
        </text>
      </g>
    </svg>
  );
}
