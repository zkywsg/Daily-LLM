const W = 700;
const H = 340;

interface Props {
  showShift: boolean;
}

const gridSize = 8;
const cellSize = 26;
const windowSize = 4;

export function ShiftedWindowDiagram({ showShift }: Props) {
  const startX1 = 60;
  const startX2 = 400;
  const startY = 60;

  function renderGrid(startX: number, shift: number, label: string, color: string) {
    return (
      <g>
        <text x={startX + gridSize * cellSize / 2} y={startY - 12} textAnchor="middle" fontSize={12} fontWeight={700} fill={color}>{label}</text>
        {Array.from({ length: gridSize }).map((_, r) =>
          Array.from({ length: gridSize }).map((_, c) => {
            const wr = Math.floor(((r + shift + gridSize) % gridSize) / windowSize);
            const wc = Math.floor(((c + shift + gridSize) % gridSize) / windowSize);
            const windowId = wr * 2 + wc;
            const colors = ["#dbeafe", "#fce7f3", "#fef3c7", "#ecfdf5"];
            return (
              <rect key={`${r}-${c}`}
                    x={startX + c * cellSize} y={startY + r * cellSize}
                    width={cellSize - 1} height={cellSize - 1}
                    fill={colors[windowId % 4]} stroke="#9ca3af" strokeWidth={0.5} />
            );
          })
        )}
        {/* window boundary lines, offset by shift */}
        {Array.from({ length: gridSize / windowSize + 1 }).map((_, i) => {
          const pos = (i * windowSize - shift + gridSize) % gridSize;
          if (pos === 0 && i > 0 && shift === 0) return null;
          return (
            <g key={i}>
              <line x1={startX + pos * cellSize} y1={startY}
                    x2={startX + pos * cellSize} y2={startY + gridSize * cellSize}
                    stroke={color} strokeWidth={2} />
              <line x1={startX} y1={startY + pos * cellSize}
                    x2={startX + gridSize * cellSize} y2={startY + pos * cellSize}
                    stroke={color} strokeWidth={2} />
            </g>
          );
        })}
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Shifted window mechanism">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Shifted Window — 交替窗口划分让边界信息流动
      </text>

      {renderGrid(startX1, 0, "W-MSA(标准)", "#3b82f6")}
      {showShift && renderGrid(startX2, 2, "SW-MSA(偏移 M/2)", "#ec4899")}

      {!showShift && (
        <text x={startX2 + 120} y={startY + 100} textAnchor="middle" fontSize={11} fill="#9ca3af">
          点按钮显示偏移窗口
        </text>
      )}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        第 n 层窗口边界在第 n+1 层窗口中央 → 原本跨边界的 patch 被合并到同一窗口 attention
      </text>
    </svg>
  );
}
