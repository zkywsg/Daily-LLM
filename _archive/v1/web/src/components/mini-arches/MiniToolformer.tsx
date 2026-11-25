import type { MiniArchProps } from "./types";

export function MiniToolformer({
  width = 160,
  height = 70,
  ariaLabel = "Toolformer 架构缩图",
}: MiniArchProps) {
  // 文本流中嵌入 [Tool(args) → result],突出"调用内联"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 文本流(token 流) — 顶部一条 */}
        <text x="80" y="12" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.65">text stream with inlined tool calls</text>
        {/* token 块串 */}
        <rect x="4" y="22" width="22" height="12" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="15" y="30.5" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.8">月球</text>
        <rect x="28" y="22" width="14" height="12" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="35" y="30.5" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.8">在</text>
        {/* 内联 tool call(高亮) */}
        <rect x="44" y="20" width="52" height="16" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="70" y="27" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.9">[QA("when?")</text>
        <text x="70" y="33" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.9">→ 1969]</text>
        <rect x="98" y="22" width="22" height="12" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="109" y="30.5" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.8">登月</text>
        <rect x="122" y="22" width="34" height="12" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="139" y="30.5" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.8">成功了</text>
        {/* 底部:perplexity filter 说明 */}
        <rect x="4" y="44" width="152" height="14" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="80" y="53" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">
          self-supervised:仅保留让 PPL ↓ 的调用
        </text>
        <text x="80" y="65" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          tool use 内化为参数,无需 prompt 教
        </text>
      </g>
    </svg>
  );
}
