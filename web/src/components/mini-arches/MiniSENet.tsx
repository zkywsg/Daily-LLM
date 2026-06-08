import type { MiniArchProps } from "./types";

export function MiniSENet({
  width = 180,
  height = 70,
  ariaLabel = "SENet 架构缩图",
}: MiniArchProps) {
  // 通道重标定：feature → GAP → FC → FC → sigmoid → × feature
  return (
    <svg
      viewBox="0 0 180 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        <rect x="0" y="22" width="30" height="32" rx="2" className="illustration__featuremap" />
        <path d="M 30 38 C 42 38, 42 12, 54 12" className="illustration__branch illustration__branch--q" fill="none" />
        {[60, 86, 112].map((x, i) => (
          <rect key={i} x={x} y="6" width="20" height="14" rx="2"
            className={i === 0 ? "illustration__block illustration__block--alt" : "illustration__proj illustration__proj--v"} />
        ))}
        <text x="70" y="16" textAnchor="middle" fontSize="7">GAP</text>
        <text x="96" y="16" textAnchor="middle" fontSize="7">FC</text>
        <text x="122" y="16" textAnchor="middle" fontSize="7">σ</text>
        <path d="M 132 12 C 140 12, 140 38, 134 38" className="illustration__branch illustration__branch--q" fill="none" />
        <text x="138" y="42" fontSize="11">×</text>
        <rect x="146" y="22" width="30" height="32" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
      </g>
    </svg>
  );
}
