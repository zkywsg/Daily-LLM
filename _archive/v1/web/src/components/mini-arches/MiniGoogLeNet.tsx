import type { MiniArchProps } from "./types";

export function MiniGoogLeNet({
  width = 160,
  height = 80,
  ariaLabel = "GoogLeNet 架构缩图",
}: MiniArchProps) {
  // Inception：4 路并行 → concat
  return (
    <svg
      viewBox="0 0 160 80"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        <rect x="0" y="32" width="14" height="14" rx="2" className="illustration__featuremap" />
        {[
          { y: 8, label: "1×1" },
          { y: 28, label: "3×3" },
          { y: 48, label: "5×5" },
          { y: 68, label: "pool" },
        ].map((br, i) => (
          <g key={i}>
            <path d={`M 14 39 C 30 39, 40 ${br.y + 6}, 56 ${br.y + 6}`} className="illustration__branch illustration__branch--q" fill="none" />
            <rect x="56" y={br.y} width="32" height="12" rx="2" className="illustration__proj illustration__proj--q" />
            <text x="72" y={br.y + 9} textAnchor="middle" fontSize="8" className="illustration__block-label illustration__block-label--small">{br.label}</text>
            <path d={`M 88 ${br.y + 6} C 100 ${br.y + 6}, 110 39, 124 39`} className="illustration__branch illustration__branch--k" fill="none" />
          </g>
        ))}
        <rect x="124" y="32" width="22" height="14" rx="2" className="illustration__featuremap illustration__featuremap--ctx" />
      </g>
    </svg>
  );
}
