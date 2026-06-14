import type { MiniArchProps } from "./types";

export function MiniSwin({
  width = 160,
  height = 70,
  ariaLabel = "Swin Transformer 架构缩图",
}: MiniArchProps) {
  // 4 个 stage 层级化:特征图逐 stage 减半,每个 stage 内 windowed attention
  // 用嵌套的方块体现"窗口划分"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 4 个 stage 的特征图,逐渐缩小 */}
        {[
          { x: 6, size: 36, win: 4, color: "ffn" },   // Stage 1: 大
          { x: 50, size: 28, win: 4, color: "ffn" },  // Stage 2
          { x: 86, size: 20, win: 2, color: "v" },    // Stage 3
          { x: 118, size: 14, win: 2, color: "v" },   // Stage 4: 小
        ].map((s, i) => {
          const cy = 35;
          const y = cy - s.size / 2;
          // 主特征图方块
          const cls = s.color === "v"
            ? "illustration__proj illustration__proj--v"
            : "illustration__proj illustration__proj--ffn";
          return (
            <g key={i}>
              <rect x={s.x} y={y} width={s.size} height={s.size} rx="1" className={cls} opacity="0.6" />
              {/* 窗口分割线 */}
              {Array.from({ length: s.win - 1 }, (_, j) => {
                const offset = ((j + 1) * s.size) / s.win;
                return (
                  <g key={j}>
                    <line x1={s.x + offset} y1={y} x2={s.x + offset} y2={y + s.size}
                      stroke="currentColor" strokeWidth="0.5" opacity="0.7" />
                    <line x1={s.x} y1={y + offset} x2={s.x + s.size} y2={y + offset}
                      stroke="currentColor" strokeWidth="0.5" opacity="0.7" />
                  </g>
                );
              })}
              {/* Stage 标签 */}
              <text x={s.x + s.size / 2} y={cy + s.size / 2 + 8}
                textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.6">
                S{i + 1}
              </text>
            </g>
          );
        })}
        {/* stage 间 patch merging 箭头 */}
        {[{ x1: 42, x2: 50 }, { x1: 78, x2: 86 }, { x1: 106, x2: 118 }].map((arr, i) => (
          <line key={i} x1={arr.x1} y1="35" x2={arr.x2} y2="35"
            className="illustration__branch illustration__branch--q" />
        ))}
        {/* 顶部标 */}
        <text x="80" y="6" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.55">
          windowed attention · 层级下采样
        </text>
      </g>
    </svg>
  );
}
