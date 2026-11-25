const W = 700;
const H = 300;

interface Props {
  strategy: "task" | "fixed";
}

const N = 16;

export function GlobalTokenDiagram({ strategy }: Props) {
  const cellW = (W - 60) / N;
  const startX = 30;
  const y = 120;

  const globalIdx = strategy === "task" ? [0, 5, 6, 7] : [0, 4, 8, 12];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Global token strategies">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Global Token — {strategy === "task" ? "任务相关(Longformer)" : "位置固定(BigBird)"}
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {strategy === "task" ? "QA 时把整个 question 设为 global(位置 5-7 高亮)" : "每隔 4 个位置固定选一个 global token"}
      </text>

      {Array.from({ length: N }).map((_, i) => {
        const isGlobal = globalIdx.includes(i);
        return (
          <g key={i}>
            {isGlobal && (
              <line x1={startX + i * cellW + cellW / 2} y1={y + 30} x2={startX + i * cellW + cellW / 2} y2={y + 70}
                    stroke="#f59e0b" strokeWidth={1} opacity={0.3} />
            )}
            <rect x={startX + i * cellW} y={y} width={cellW - 2} height={30} rx={2}
                  fill={isGlobal ? "#fef3c7" : "#f3f4f6"}
                  stroke={isGlobal ? "#f59e0b" : "#d1d5db"} strokeWidth={isGlobal ? 2 : 1} />
            {isGlobal && (
              <text x={startX + i * cellW + cellW / 2} y={y - 8} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">global</text>
            )}
          </g>
        );
      })}

      {/* cross connections from global to all */}
      {globalIdx.map((g) => (
        Array.from({ length: N }).map((_, j) => (
          <line key={`${g}-${j}`} x1={startX + g * cellW + cellW / 2} y1={y}
                x2={startX + j * cellW + cellW / 2} y2={y - 20}
                stroke="#f59e0b" strokeWidth={0.5} opacity={0.15} />
        ))
      ))}

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={11} fontWeight={600} fill="#92400e">
        Global token 对所有位置做 dense · 所有位置也 attend 到它
      </text>
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        复杂度 O(N × g) = O(N)(g ≈ 8-16 个)
      </text>
    </svg>
  );
}
