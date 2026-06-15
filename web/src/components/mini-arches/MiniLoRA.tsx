import type { MiniArchProps } from "./types";

export function MiniLoRA({
  width = 160,
  height = 70,
  ariaLabel = "LoRA 架构缩图",
}: MiniArchProps) {
  // h = W₀x + BA x;突出 W₀ 冻结 + 低秩 BA 旁路
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* x */}
        <rect x="2" y="30" width="14" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="9" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">x</text>
        {/* 主路:W₀ 冻结 */}
        <rect x="32" y="10" width="60" height="14" rx="2"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__block" />
        <text x="62" y="16" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">W₀  d×d</text>
        <text x="62" y="21" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.55">❄ frozen</text>
        {/* 旁路:A 矩阵(d × r,窄高) */}
        <rect x="32" y="44" width="22" height="20" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="43" y="52" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">A</text>
        <text x="43" y="59" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">r×d</text>
        <text x="43" y="63" textAnchor="middle" fontSize="3.4"
          fill="currentColor" opacity="0.6">(r=8)</text>
        {/* B 矩阵 */}
        <rect x="62" y="44" width="30" height="20" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="77" y="52" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">B</text>
        <text x="77" y="59" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">d×r</text>
        <text x="77" y="63" textAnchor="middle" fontSize="3.4"
          fill="currentColor" opacity="0.6">init=0</text>
        {/* + */}
        <circle cx="104" cy="35" r="6"
          className="illustration__proj illustration__proj--v" />
        <text x="104" y="38" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">+</text>
        {/* h */}
        <rect x="120" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="127" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">h</text>
        {/* 箭头:x → 主路 W₀ */}
        <line x1="16" y1="35" x2="22" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="22" y1="35" x2="22" y2="17"
          className="illustration__branch illustration__branch--q" />
        <line x1="22" y1="17" x2="32" y2="17"
          className="illustration__branch illustration__branch--q" />
        <line x1="92" y1="17" x2="104" y2="29"
          className="illustration__branch illustration__branch--v" />
        {/* 旁路:x → A → B */}
        <line x1="22" y1="35" x2="22" y2="54"
          className="illustration__branch illustration__branch--q" />
        <line x1="22" y1="54" x2="32" y2="54"
          className="illustration__branch illustration__branch--q" />
        <line x1="54" y1="54" x2="62" y2="54"
          className="illustration__branch illustration__branch--q" />
        <line x1="92" y1="54" x2="104" y2="41"
          className="illustration__branch illustration__branch--v" />
        {/* + → h */}
        <line x1="110" y1="35" x2="120" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="6" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.7">h = W₀x + BAx · 推理可合并</text>
      </g>
    </svg>
  );
}
