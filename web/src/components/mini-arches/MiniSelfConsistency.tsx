import type { MiniArchProps } from "./types";

export function MiniSelfConsistency({
  width = 160,
  height = 70,
  ariaLabel = "Self-Consistency 架构缩图",
}: MiniArchProps) {
  // Q → 多条 CoT 路径(并行)→ vote → A
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Q */}
        <rect x="4" y="28" width="18" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="13" y="38" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.8">Q</text>
        {/* 三条并行路径(reasoning paths) */}
        <rect x="36" y="6" width="58" height="11" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="65" y="14" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.75">path₁ → 42 ✓</text>
        <rect x="36" y="29" width="58" height="11" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="65" y="37" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.75">path₂ → 42 ✓</text>
        <rect x="36" y="52" width="58" height="11" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.5" />
        <text x="65" y="60" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">path₃ → 37 ✗</text>
        {/* 分叉箭头 */}
        <line x1="22" y1="35" x2="36" y2="11"
          className="illustration__branch illustration__branch--q" />
        <line x1="22" y1="35" x2="36" y2="34"
          className="illustration__branch illustration__branch--q" />
        <line x1="22" y1="35" x2="36" y2="57"
          className="illustration__branch illustration__branch--q" />
        {/* vote 块 */}
        <rect x="108" y="22" width="22" height="26" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="119" y="34" textAnchor="middle" fontSize="5.5" fontWeight="600"
          fill="currentColor" opacity="0.85">vote</text>
        <text x="119" y="42" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">majority</text>
        {/* 三条聚合到 vote */}
        <line x1="94" y1="11" x2="108" y2="29"
          className="illustration__branch illustration__branch--v" />
        <line x1="94" y1="34" x2="108" y2="35"
          className="illustration__branch illustration__branch--v" />
        <line x1="94" y1="57" x2="108" y2="41"
          className="illustration__branch illustration__branch--v" />
        {/* A */}
        <rect x="138" y="28" width="18" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="147" y="38" textAnchor="middle" fontSize="7" fontWeight="600"
          fill="currentColor" opacity="0.85">42</text>
        <line x1="130" y1="35" x2="138" y2="35"
          className="illustration__branch illustration__branch--v" />
      </g>
    </svg>
  );
}
