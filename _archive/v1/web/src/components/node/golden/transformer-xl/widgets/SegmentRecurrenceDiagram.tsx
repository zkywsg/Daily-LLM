interface Props {
  currentSegment: number; // 1-based,当前正在处理的段
  totalSegments?: number;
}

const W = 700;
const H = 300;

export function SegmentRecurrenceDiagram({ currentSegment, totalSegments = 4 }: Props) {
  const PAD_L = 40;
  const PAD_T = 60;
  const segW = 130;
  const gap = 30;
  const boxH = 60;

  const xOf = (seg: number) => PAD_L + (seg - 1) * (segW + gap);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Segment-level recurrence 数据流">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        当前处理第 {currentSegment} 段 — 蓝色实线为当前段,灰色虚线为已缓存(stop-gradient)的历史段
      </text>

      {Array.from({ length: totalSegments }, (_, i) => i + 1).map((seg) => {
        const x = xOf(seg);
        const isCurrent = seg === currentSegment;
        const isCached = seg < currentSegment;
        const isFuture = seg > currentSegment;
        const fill = isCurrent ? "#dbeafe" : isCached ? "#f3f4f6" : "#ffffff";
        const stroke = isCurrent ? "#3b82f6" : isCached ? "#9ca3af" : "#e5e7eb";
        const dash = isCached ? "4 3" : undefined;

        return (
          <g key={seg} opacity={isFuture ? 0.35 : 1}>
            <rect
              x={x}
              y={PAD_T}
              width={segW}
              height={boxH}
              fill={fill}
              stroke={stroke}
              strokeWidth={isCurrent ? 2 : 1.4}
              strokeDasharray={dash}
              rx={6}
            />
            <text x={x + segW / 2} y={PAD_T + boxH / 2 - 4} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              段 {seg}
            </text>
            <text x={x + segW / 2} y={PAD_T + boxH / 2 + 14} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              {isCurrent ? "当前(Q 来源)" : isCached ? "已缓存 SG(·)" : "未处理"}
            </text>

            {/* 段间箭头 */}
            {seg < totalSegments && (
              <line
                x1={x + segW}
                y1={PAD_T + boxH / 2}
                x2={x + segW + gap}
                y2={PAD_T + boxH / 2}
                stroke="#9ca3af"
                strokeWidth={1.5}
                markerEnd="url(#arrow-seg)"
              />
            )}

            {/* 当前段向前一段的 K/V attend 曲线 */}
            {isCurrent && seg > 1 && (
              <path
                d={`M ${x + 10} ${PAD_T} Q ${x - gap / 2} ${PAD_T - 30} ${xOf(seg - 1) + segW - 10} ${PAD_T}`}
                fill="none"
                stroke="#3b82f6"
                strokeWidth={1.4}
                strokeDasharray="3 2"
                markerEnd="url(#arrow-attend)"
              />
            )}
          </g>
        );
      })}

      <defs>
        <marker id="arrow-seg" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#9ca3af" />
        </marker>
        <marker id="arrow-attend" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#3b82f6" />
        </marker>
      </defs>

      <text x={W / 2} y={PAD_T + boxH + 40} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        当前段的 Q 只来自本段,K/V = [SG(上一段隐状态) ; 本段隐状态] — 信息单向流动,梯度不跨段回传
      </text>
      <text x={W / 2} y={PAD_T + boxH + 58} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        每层独立缓存、逐层向上累积 — 理论最大上下文 O(段数 × 层数 × 段长)
      </text>
    </svg>
  );
}
