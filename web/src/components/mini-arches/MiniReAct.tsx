import type { MiniArchProps } from "./types";

export function MiniReAct({
  width = 160,
  height = 70,
  ariaLabel = "ReAct 架构缩图",
}: MiniArchProps) {
  // LLM ⇄ tools 循环:Thought → Action → Observation → Thought ...
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧 LLM 大块 */}
        <rect x="6" y="14" width="36" height="42" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="24" y="30" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.85">LLM</text>
        <text x="24" y="40" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">Thought</text>
        {/* 中间循环箭头 + 标签 */}
        <text x="55" y="20" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.75">Action →</text>
        <text x="55" y="55" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.75">← Obs</text>
        <line x1="42" y1="24" x2="68" y2="24"
          className="illustration__branch illustration__branch--q" />
        <line x1="68" y1="50" x2="42" y2="50"
          className="illustration__branch illustration__branch--v" />
        {/* 右侧工具集 */}
        <rect x="74" y="6" width="56" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="102" y="15" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.8">🔍 search</text>
        <rect x="74" y="22" width="56" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="102" y="31" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.8">🧮 calc</text>
        <rect x="74" y="38" width="56" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="102" y="47" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.8">🐍 python</text>
        <rect x="74" y="54" width="56" height="10" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.5" />
        <text x="102" y="61" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.65">...</text>
        {/* 右侧到 final */}
        <rect x="138" y="22" width="20" height="26" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="148" y="32" textAnchor="middle" fontSize="5.5" fontWeight="600"
          fill="currentColor" opacity="0.85">Final</text>
        <text x="148" y="42" textAnchor="middle" fontSize="5.5" fontWeight="600"
          fill="currentColor" opacity="0.85">A</text>
        <line x1="42" y1="35" x2="138" y2="35"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--v" />
      </g>
    </svg>
  );
}
