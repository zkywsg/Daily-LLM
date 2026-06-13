import type { MiniArchProps } from "./types";

export function MiniSparseAttention({
  width = 160,
  height = 70,
  ariaLabel = "Sparse Attention 架构缩图",
}: MiniArchProps) {
  // N×N attention 矩阵的稀疏 mask 模式可视化:
  // - 主对角带(局部滑窗)
  // - 一行 + 一列填满(global token)
  const cell = 5;
  const N = 12;
  const origin = { x: 14, y: 5 };
  const cells: { r: number; c: number; type: "local" | "global" }[] = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const isLocal = Math.abs(r - c) <= 1;
      const isGlobal = r === 0 || c === 0; // 第一个 token 作为 global
      if (isLocal || isGlobal) {
        cells.push({ r, c, type: isGlobal ? "global" : "local" });
      }
    }
  }
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 矩阵网格底图(浅色) */}
        <rect
          x={origin.x}
          y={origin.y}
          width={N * cell}
          height={N * cell}
          fill="none"
          className="illustration__block"
        />
        {/* 稀疏激活的格子 */}
        {cells.map((c, i) => (
          <rect
            key={i}
            x={origin.x + c.c * cell + 0.5}
            y={origin.y + c.r * cell + 0.5}
            width={cell - 1}
            height={cell - 1}
            className={
              c.type === "global"
                ? "illustration__proj illustration__proj--v"
                : "illustration__proj illustration__proj--ffn"
            }
          />
        ))}
        {/* 右侧:N 个序列 token 一字排开,呼应矩阵的行/列含义 */}
        {Array.from({ length: N }, (_, i) => (
          <rect
            key={`t-${i}`}
            x={88 + i * 5.5}
            y="32"
            width="4"
            height="6"
            rx="1"
            className={i === 0 ? "illustration__featuremap illustration__featuremap--ctx" : "illustration__layer illustration__layer--input"}
          />
        ))}
      </g>
    </svg>
  );
}
