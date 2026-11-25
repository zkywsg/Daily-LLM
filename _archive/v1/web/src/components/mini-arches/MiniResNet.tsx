import type { MiniArchProps } from "./types";

export function MiniResNet({
  width = 160,
  height = 70,
  ariaLabel = "ResNet 架构缩图",
}: MiniArchProps) {
  // 残差：旁路 skip arc
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        <rect x="0" y="32" width="14" height="14" rx="2" className="illustration__featuremap" />
        {[30, 60, 90].map((x, i) => (
          <rect key={i} x={x} y="32" width="20" height="14" rx="2" className="illustration__proj illustration__proj--ffn" />
        ))}
        <circle cx="125" cy="39" r="6" className="illustration__addnorm" />
        <text x="125" y="42" textAnchor="middle" fontSize="10">⊕</text>
        <rect x="140" y="32" width="14" height="14" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
        {/* skip 弧 */}
        <path d="M 7 32 C 7 6, 125 6, 125 32" className="illustration__residual" fill="none" />
      </g>
    </svg>
  );
}
