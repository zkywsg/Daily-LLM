import type { MiniArchProps } from "./types";

export function MiniInstructGPT({
  width = 160,
  height = 70,
  ariaLabel = "InstructGPT 架构缩图",
}: MiniArchProps) {
  // 中央大 LLM 块(GPT-3 175B) + 左侧 instruction prompts + 右侧
  // aligned output;突出"指令 → 对齐回答"的范式
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧 instruction prompts(多种任务类型) */}
        {[10, 22, 34, 46].map((y, i) => (
          <rect
            key={i}
            x="4"
            y={y}
            width="20"
            height="8"
            rx="1"
            className="illustration__layer illustration__layer--input"
            opacity={0.9 - i * 0.12}
          />
        ))}
        <text
          x="14"
          y="62"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.6"
        >
          指令
        </text>
        {/* 中央 InstructGPT 模型 — 大块且带"调校"标记 */}
        <rect
          x="42"
          y="14"
          width="76"
          height="42"
          rx="4"
          className="illustration__proj illustration__proj--v"
        />
        <text
          x="80"
          y="32"
          textAnchor="middle"
          fontSize="8"
          fontWeight="600"
          fill="currentColor"
          opacity="0.85"
        >
          GPT-3
        </text>
        <text
          x="80"
          y="42"
          textAnchor="middle"
          fontSize="7"
          fill="currentColor"
          opacity="0.7"
        >
          + RLHF
        </text>
        {/* 三阶段标记(右上小图标) */}
        {[50, 56, 62].map((x, i) => (
          <rect
            key={i}
            x={x}
            y="18"
            width="3"
            height="3"
            rx="0.5"
            fill="currentColor"
            opacity="0.5"
          />
        ))}
        {/* 右侧 aligned output(单条但更"高质量") */}
        <rect
          x="136"
          y="30"
          width="20"
          height="10"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <text
          x="146"
          y="62"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.6"
        >
          对齐回答
        </text>
        {/* 输入/输出箭头 */}
        <line
          x1="24"
          y1="32"
          x2="42"
          y2="32"
          className="illustration__branch illustration__branch--q"
        />
        <line
          x1="118"
          y1="35"
          x2="136"
          y2="35"
          className="illustration__branch illustration__branch--v"
        />
      </g>
    </svg>
  );
}
