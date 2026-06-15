import type { MiniArchProps } from "./types";

export function MiniDeepSeekR1({
  width = 160,
  height = 70,
  ariaLabel = "DeepSeek-R1 架构缩图",
}: MiniArchProps) {
  // Q → 一组采样 response (G 个) → group baseline → policy update
  // 突出 GRPO:group sampling 替代 value model,推理 trace 公开
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
        <rect x="4" y="28" width="16" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="12" y="38" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.8">Q</text>
        {/* G 个 group 采样 — 4 条平行 response */}
        <rect x="28" y="6" width="56" height="9" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="56" y="12.5" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.78">y₁  r=1</text>
        <rect x="28" y="19" width="56" height="9" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="56" y="25.5" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.78">y₂  r=1</text>
        <rect x="28" y="32" width="56" height="9" rx="1.5"
          className="illustration__proj illustration__proj--ffn" opacity="0.5" />
        <text x="56" y="38.5" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.6">y₃  r=0</text>
        <rect x="28" y="45" width="56" height="9" rx="1.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="56" y="51.5" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.78">y₄  r=1</text>
        {/* Q → 4 个 */}
        <line x1="20" y1="35" x2="28" y2="10"
          className="illustration__branch illustration__branch--q" />
        <line x1="20" y1="35" x2="28" y2="23"
          className="illustration__branch illustration__branch--q" />
        <line x1="20" y1="35" x2="28" y2="36"
          className="illustration__branch illustration__branch--q" />
        <line x1="20" y1="35" x2="28" y2="49"
          className="illustration__branch illustration__branch--q" />
        {/* group baseline 块 */}
        <rect x="94" y="14" width="28" height="32" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="108" y="24" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.8">group</text>
        <text x="108" y="32" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">mean μ</text>
        <text x="108" y="40" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.65">no V(x)</text>
        {/* 4 → group */}
        <line x1="84" y1="10" x2="94" y2="20"
          className="illustration__branch illustration__branch--v" />
        <line x1="84" y1="23" x2="94" y2="28"
          className="illustration__branch illustration__branch--v" />
        <line x1="84" y1="36" x2="94" y2="34"
          className="illustration__branch illustration__branch--v" />
        <line x1="84" y1="49" x2="94" y2="40"
          className="illustration__branch illustration__branch--v" />
        {/* policy 块 */}
        <rect x="128" y="22" width="28" height="16" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="142" y="29" textAnchor="middle" fontSize="5.5" fontWeight="600"
          fill="currentColor" opacity="0.85">π_θ</text>
        <text x="142" y="36" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">update</text>
        <line x1="122" y1="30" x2="128" y2="30"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="63" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">
          GRPO · rule-based reward · 开源 reasoning trace
        </text>
      </g>
    </svg>
  );
}
