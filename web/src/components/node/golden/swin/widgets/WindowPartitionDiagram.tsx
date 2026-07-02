const W = 700;
const H = 320;

interface Props {
  mode: "full" | "windowed";
}

// 8x8 grid of patches, 用来演示 full attention (全连接) vs windowed (7x7 局部,用小格模拟窗口边界)
export function WindowPartitionDiagram({ mode }: Props) {
  const gridSize = 8;
  const cellSize = 28;
  const startX = (W - gridSize * cellSize) / 2;
  const startY = 60;
  const windowSize = 4; // 演示用 4x4 (代表实际 7x7)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Full attention vs windowed attention">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "full" ? "全局 Attention — 每个 patch 看所有 patch" : "Windowed Attention — 每个 patch 只看窗口内"}
      </text>

      {/* grid cells */}
      {Array.from({ length: gridSize }).map((_, r) =>
        Array.from({ length: gridSize }).map((_, c) => {
          const inSameWindow = mode === "windowed" &&
            Math.floor(r / windowSize) === Math.floor(3 / windowSize) &&
            Math.floor(c / windowSize) === Math.floor(3 / windowSize);
          const isCenter = r === 3 && c === 3;
          return (
            <rect key={`${r}-${c}`}
                  x={startX + c * cellSize} y={startY + r * cellSize}
                  width={cellSize - 2} height={cellSize - 2}
                  fill={isCenter ? "#ec4899" : inSameWindow ? "#fce7f3" : mode === "full" ? "#dbeafe" : "#f3f4f6"}
                  stroke={isCenter ? "#831843" : "#9ca3af"} strokeWidth={isCenter ? 2 : 0.6} />
          );
        })
      )}

      {/* window boundaries */}
      {mode === "windowed" && Array.from({ length: gridSize / windowSize + 1 }).map((_, i) => (
        <g key={i}>
          <line x1={startX + i * windowSize * cellSize} y1={startY}
                x2={startX + i * windowSize * cellSize} y2={startY + gridSize * cellSize}
                stroke="#f59e0b" strokeWidth={2} />
          <line x1={startX} y1={startY + i * windowSize * cellSize}
                x2={startX + gridSize * cellSize} y2={startY + i * windowSize * cellSize}
                stroke="#f59e0b" strokeWidth={2} />
        </g>
      ))}

      {/* connections from center */}
      {mode === "full" && Array.from({ length: gridSize }).map((_, r) =>
        Array.from({ length: gridSize }).map((_, c) => {
          if (r === 3 && c === 3) return null;
          return (
            <line key={`l-${r}-${c}`}
                  x1={startX + 3 * cellSize + cellSize / 2} y1={startY + 3 * cellSize + cellSize / 2}
                  x2={startX + c * cellSize + cellSize / 2} y2={startY + r * cellSize + cellSize / 2}
                  stroke="#3b82f6" strokeWidth={0.4} opacity={0.3} />
          );
        })
      )}

      <text x={W / 2} y={H - 24} textAnchor="middle" fontSize={11} fontWeight={700} fill={mode === "full" ? "#1e40af" : "#92400e"}>
        {mode === "full" ? "8×8=64 个 patch,attention = 64² = 4096 次连接" : "窗口大小 4×4,粉色中心只连自己窗口内 16 个 patch"}
      </text>
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        真实 Swin 用 7×7 窗口(此处简化为 4×4 便于展示)
      </text>
    </svg>
  );
}
