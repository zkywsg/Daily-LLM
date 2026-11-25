import type { MiniArchProps } from "./types";

export function MiniBLIP({
  width = 160,
  height = 70,
  ariaLabel = "BLIP / BLIP-2 架构缩图",
}: MiniArchProps) {
  // BLIP-2 风格:冻结 ViT + Q-Former + 冻结 LLM
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:冻结的 ViT */}
        <rect x="4" y="14" width="28" height="42" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.55" />
        <text x="18" y="32" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.7">ViT</text>
        <text x="18" y="40" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.6">frozen</text>
        {/* 中央:Q-Former + 32 Q tokens */}
        <rect x="50" y="18" width="44" height="34" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="72" y="30" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">Q-Former</text>
        <text x="72" y="38" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">32 Q tokens</text>
        <text x="72" y="46" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">188M ✓</text>
        {/* 右侧:冻结的 LLM */}
        <rect x="112" y="14" width="44" height="42" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.55" />
        <text x="134" y="32" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.7">LLM</text>
        <text x="134" y="40" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.6">frozen</text>
        {/* 连接箭头 */}
        <line x1="32" y1="35" x2="50" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="94" y1="35" x2="112" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部:1.5% 可训练 */}
        <text x="80" y="65" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">
          冻结视觉 + LLM · 仅 1.5% 可训练
        </text>
      </g>
    </svg>
  );
}
