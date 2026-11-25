import type { MiniArchProps } from "./types";

export function MiniCoT({
  width = 160,
  height = 70,
  ariaLabel = "Chain-of-Thought 架构缩图",
}: MiniArchProps) {
  // Q → step1 → step2 → step3 → A,突出"中间步骤显式输出"
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
        <rect x="4" y="22" width="18" height="16" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="13" y="33" textAnchor="middle" fontSize="7" fontWeight="600"
          fill="currentColor" opacity="0.8">Q</text>
        {/* step 1 */}
        <rect x="30" y="22" width="22" height="16" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="41" y="32" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.8">step₁</text>
        {/* step 2 */}
        <rect x="58" y="22" width="22" height="16" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="69" y="32" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.8">step₂</text>
        {/* step 3 */}
        <rect x="86" y="22" width="22" height="16" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="97" y="32" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.8">step₃</text>
        {/* A */}
        <rect x="116" y="22" width="40" height="16" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="136" y="33" textAnchor="middle" fontSize="7" fontWeight="600"
          fill="currentColor" opacity="0.85">A</text>
        {/* 箭头 */}
        <line x1="22" y1="30" x2="30" y2="30"
          className="illustration__branch illustration__branch--q" />
        <line x1="52" y1="30" x2="58" y2="30"
          className="illustration__branch illustration__branch--q" />
        <line x1="80" y1="30" x2="86" y2="30"
          className="illustration__branch illustration__branch--q" />
        <line x1="108" y1="30" x2="116" y2="30"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="58" textAnchor="middle" fontSize="5.5"
          fill="currentColor" opacity="0.6">
          "let's think step by step"
        </text>
        <text x="80" y="65" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.5">
          prompt 触发 · 显式中间步骤
        </text>
      </g>
    </svg>
  );
}
