import type { MiniArchProps } from "./types";

export function MiniGloVe({
  width = 160,
  height = 70,
  ariaLabel = "GloVe 架构缩图",
}: MiniArchProps) {
  // 全局共现矩阵 X_ij → 加权矩阵分解
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:共现矩阵 X_ij 网格 */}
        <text x="22" y="6" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">co-occur X_ij</text>
        <g transform="translate(4, 10)">
          {/* 7×7 网格,色彩深浅代表 log X_ij */}
          {Array.from({ length: 49 }).map((_, idx) => {
            const r = Math.floor(idx / 7);
            const c = idx % 7;
            // 对角线和近邻颜色更深(常见共现)
            const dist = Math.abs(r - c);
            const op = dist === 0 ? 0.9 : dist === 1 ? 0.7 : dist === 2 ? 0.5 : 0.3;
            return (
              <rect key={idx}
                x={c * 5.2} y={r * 5.2}
                width="4.5" height="4.5" rx="0.4"
                className="illustration__featuremap illustration__featuremap--ctx"
                opacity={op} />
            );
          })}
        </g>
        {/* ≈ 号 */}
        <text x="52" y="36" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">≈</text>
        {/* W(d×r 矩阵) */}
        <g transform="translate(60, 10)">
          <text x="14" y="-4" textAnchor="middle" fontSize="4.5" fontWeight="600"
            fill="currentColor" opacity="0.75">W</text>
          {Array.from({ length: 21 }).map((_, idx) => {
            const r = Math.floor(idx / 3);
            const c = idx % 3;
            return (
              <rect key={idx}
                x={c * 5} y={r * 5.2}
                width="4.3" height="4.5" rx="0.4"
                className="illustration__proj illustration__proj--ffn"
                opacity={0.7} />
            );
          })}
        </g>
        {/* × */}
        <text x="82" y="36" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">·</text>
        {/* W~ᵀ(r×d 矩阵,横长) */}
        <g transform="translate(90, 24)">
          <text x="18" y="-4" textAnchor="middle" fontSize="4.5" fontWeight="600"
            fill="currentColor" opacity="0.75">W̃ᵀ</text>
          {Array.from({ length: 21 }).map((_, idx) => {
            const r = Math.floor(idx / 7);
            const c = idx % 7;
            return (
              <rect key={idx}
                x={c * 5} y={r * 5}
                width="4.3" height="4.3" rx="0.4"
                className="illustration__proj illustration__proj--v"
                opacity={0.7} />
            );
          })}
        </g>
        {/* loss + weighted */}
        <rect x="128" y="20" width="30" height="26" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="143" y="28" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.8">f(X_ij)</text>
        <text x="143" y="34" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">·(wᵀw̃ − log X)²</text>
        <text x="143" y="40" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.65">weighted</text>
        {/* 底部 */}
        <text x="80" y="68" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          全局共现统计 · count-based · 加权 log-bilinear
        </text>
      </g>
    </svg>
  );
}
