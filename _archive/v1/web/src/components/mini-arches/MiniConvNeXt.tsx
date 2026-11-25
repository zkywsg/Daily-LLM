import type { MiniArchProps } from "./types";

export function MiniConvNeXt({
  width = 160,
  height = 70,
  ariaLabel = "ConvNeXt 架构缩图",
}: MiniArchProps) {
  // 现代化残差块：一个 7×7 大核 depthwise 宽块 + 两个 1×1 细块 + skip 弧
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        <rect x="2" y="32" width="14" height="14" rx="2"
          className="illustration__featuremap" />
        {/* 7×7 depthwise 大核块（更宽、更圆角） */}
        <rect x="26" y="30" width="38" height="18" rx="6"
          className="illustration__proj illustration__proj--v" />
        {/* 两个 1×1 pointwise 细块 */}
        <rect x="72" y="32" width="12" height="14" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <rect x="90" y="32" width="12" height="14" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <circle cx="118" cy="39" r="6" className="illustration__addnorm" />
        <text x="118" y="42" textAnchor="middle" fontSize="10">⊕</text>
        <rect x="134" y="32" width="14" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        {/* skip 弧 */}
        <path d="M 9 32 C 9 8, 118 8, 118 33" className="illustration__residual" fill="none" />
      </g>
    </svg>
  );
}
