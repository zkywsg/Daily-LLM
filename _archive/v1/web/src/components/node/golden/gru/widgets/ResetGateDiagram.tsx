const W = 700;
const H = 320;

interface Props {
  r: number; // 0..1
}

function Box({ x, y, w, h, fill, stroke, label, sub }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function ResetGateDiagram({ r }: Props) {
  const historyOpacity = 0.2 + r * 0.8;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GRU reset gate diagram">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        重置门 r — 决定候选状态用多少历史(r = {r.toFixed(2)})
      </text>

      {/* h_prev */}
      <Box x={30} y={80} w={110} h={40} fill="#dbeafe" stroke="#3b82f6" label="h_{t-1}" />

      {/* x_t */}
      <Box x={30} y={160} w={110} h={40} fill="#fce7f3" stroke="#ec4899" label="x_t" />

      {/* r gate */}
      <Arrow x1={140} y1={100} x2={200} y2={100} color="#3b82f6" id="r-a1" />
      <Arrow x1={140} y1={180} x2={200} y2={130} color="#ec4899" id="r-a2" />
      <Box x={200} y={90} w={90} h={40} fill="#fef3c7" stroke="#f59e0b" label="r_t = σ(...)" sub="重置门" />

      {/* r * h_prev, opacity 表示保留多少历史 */}
      <Arrow x1={140} y1={100} x2={330} y2={210} color="#3b82f6" id="r-a3" />
      <Arrow x1={290} y1={110} x2={330} y2={190} color="#f59e0b" id="r-a4" />
      <rect x={330} y={190} width={140} height={40} rx={4}
            fill="#dbeafe" fillOpacity={historyOpacity} stroke="#3b82f6" strokeWidth={1.4} />
      <text x={400} y={214} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">
        r ⊙ h_{"{t-1}"}
      </text>

      {/* x_t → candidate */}
      <Arrow x1={140} y1={180} x2={330} y2={100} color="#ec4899" id="r-a5" />

      {/* candidate h_cand */}
      <Arrow x1={470} y1={210} x2={530} y2={150} color="#3b82f6" id="r-a6" />
      <Arrow x1={140} y1={190} x2={530} y2={130} color="#ec4899" id="r-a7" />
      <Box x={530} y={110} w={130} h={50} fill="#ecfdf5" stroke="#10b981" label="h̃_t = tanh(...)" sub="候选状态" />

      {/* r=0/1 说明 */}
      <text x={400} y={260} textAnchor="middle" fontSize={11} fontWeight={700} fill={r < 0.3 ? "#831843" : r > 0.7 ? "#065f46" : "#92400e"}>
        {r < 0.3
          ? "r ≈ 0:完全忽略历史,当前时刻视为新序列起点"
          : r > 0.7
          ? "r ≈ 1:完全使用历史,正常延续上下文"
          : "r 中等:部分保留历史"}
      </text>

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        r 只影响候选 h̃ 的计算(新写入内容)· 不直接作用在 h_t 上(保留的历史)
      </text>
    </svg>
  );
}
