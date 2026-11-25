import type { MiniArchProps } from "./types";

export function MiniCLIP({
  width = 160,
  height = 70,
  ariaLabel = "CLIP 架构缩图",
}: MiniArchProps) {
  // 双塔:左侧图像编码器 + 右侧文本编码器,中间相似度矩阵 + 对角线高亮
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左塔:图像 + ViT */}
        <rect x="4" y="8" width="18" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="13" y="17" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.6">img</text>
        <rect x="4" y="26" width="18" height="34" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="13" y="46" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">ViT</text>
        {/* 右塔:文本 + Transformer */}
        <rect x="138" y="8" width="18" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="147" y="17" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.6">txt</text>
        <rect x="138" y="26" width="18" height="34" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="147" y="46" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">Trf</text>
        {/* 中央:N×N 相似度矩阵 */}
        {(() => {
          const N = 6;
          const cellSize = 5;
          const ox = 65, oy = 18;
          const cells = [];
          for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
              const isDiag = r === c;
              cells.push(
                <rect key={`${r}-${c}`}
                  x={ox + c * cellSize + 0.3}
                  y={oy + r * cellSize + 0.3}
                  width={cellSize - 0.6}
                  height={cellSize - 0.6}
                  className={isDiag
                    ? "illustration__featuremap illustration__featuremap--ctx"
                    : "illustration__proj illustration__proj--ffn"}
                  opacity={isDiag ? 1 : 0.25}
                />
              );
            }
          }
          return cells;
        })()}
        {/* 双塔到矩阵的连接 */}
        <line x1="22" y1="36" x2="65" y2="33"
          className="illustration__branch illustration__branch--q" />
        <line x1="138" y1="36" x2="95" y2="33"
          className="illustration__branch illustration__branch--v" />
        {/* 底部:对比学习 */}
        <text x="80" y="64" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">
          对比学习 · 对角线 ↑ 其他 ↓
        </text>
      </g>
    </svg>
  );
}
