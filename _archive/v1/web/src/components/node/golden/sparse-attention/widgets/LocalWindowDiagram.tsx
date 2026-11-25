const W = 700;
const H = 240;

interface Props {
  windowSize: number; // 2..10 (演示用缩小尺度)
  centerIdx: number;
}

const N = 20;

export function LocalWindowDiagram({ windowSize, centerIdx }: Props) {
  const cellW = (W - 60) / N;
  const startX = 30;
  const y = 100;
  const w = Math.floor(windowSize / 2);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Local sliding window on 1D sequence">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Local 滑窗 — 每个位置只看周围 w={windowSize} 个邻居
      </text>

      {Array.from({ length: N }).map((_, i) => {
        const inWindow = Math.abs(i - centerIdx) <= w;
        const isCenter = i === centerIdx;
        return (
          <g key={i}>
            <rect x={startX + i * cellW} y={y} width={cellW - 2} height={30} rx={2}
                  fill={isCenter ? "#fce7f3" : inWindow ? "#dbeafe" : "#f3f4f6"}
                  stroke={isCenter ? "#ec4899" : inWindow ? "#3b82f6" : "#d1d5db"}
                  strokeWidth={isCenter ? 2 : 1} />
            {isCenter && (
              <text x={startX + i * cellW + cellW / 2} y={y - 8} textAnchor="middle" fontSize={9} fontWeight={700} fill="#ec4899">query</text>
            )}
          </g>
        );
      })}

      <text x={W / 2} y={y + 60} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">
        位置 {centerIdx} 只 attend 到 [{Math.max(0, centerIdx - w)}, {Math.min(N - 1, centerIdx + w)}] 共 {Math.min(N - 1, centerIdx + w) - Math.max(0, centerIdx - w) + 1} 个位置
      </text>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        w=512 是经验最优 — 太小丢中距依赖,太大接近 dense 失去稀疏化意义
      </text>
    </svg>
  );
}
