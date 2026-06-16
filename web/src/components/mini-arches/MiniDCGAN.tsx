import type { MiniArchProps } from "./types";

export function MiniDCGAN({
  width = 160,
  height = 70,
  ariaLabel = "DCGAN 架构缩图",
}: MiniArchProps) {
  // ConvT 金字塔上采样 + Conv 金字塔下采样,对称
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* G:从 1×1×z 上采样到 64×64 — 倒金字塔 */}
        <text x="40" y="6" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">G (ConvT ↑)</text>
        <rect x="6" y="32" width="6" height="6" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="9" y="44" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">1²</text>
        <rect x="16" y="28" width="10" height="14" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="21" y="48" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">4²</text>
        <rect x="30" y="24" width="14" height="22" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="37" y="52" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">16²</text>
        <rect x="48" y="18" width="20" height="34" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="58" y="58" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">64²</text>
        {/* 箭头 G */}
        <line x1="12" y1="35" x2="16" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="26" y1="35" x2="30" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="35" x2="48" y2="35"
          className="illustration__branch illustration__branch--q" />
        {/* 分隔:G→image→D */}
        <text x="78" y="36" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">→</text>
        {/* D:从 64×64 下采样到 1 — 正金字塔 */}
        <text x="118" y="6" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">D (Conv ↓)</text>
        <rect x="86" y="18" width="20" height="34" rx="0.5"
          className="illustration__proj illustration__proj--ffn" />
        <text x="96" y="58" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">64²</text>
        <rect x="110" y="24" width="14" height="22" rx="0.5"
          className="illustration__proj illustration__proj--ffn" />
        <text x="117" y="52" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">16²</text>
        <rect x="128" y="28" width="10" height="14" rx="0.5"
          className="illustration__proj illustration__proj--ffn" />
        <text x="133" y="48" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">4²</text>
        <rect x="142" y="32" width="6" height="6" rx="0.5"
          className="illustration__proj illustration__proj--v" />
        <text x="145" y="44" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.6">1</text>
        {/* 箭头 D */}
        <line x1="106" y1="35" x2="110" y2="35"
          className="illustration__branch illustration__branch--v" />
        <line x1="124" y1="35" x2="128" y2="35"
          className="illustration__branch illustration__branch--v" />
        <line x1="138" y1="35" x2="142" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="68" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          CNN + BatchNorm + Adam(2e-4, β₁=0.5)
        </text>
      </g>
    </svg>
  );
}
