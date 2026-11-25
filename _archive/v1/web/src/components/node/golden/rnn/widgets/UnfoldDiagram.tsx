const W = 700;
const H = 320;

interface Props {
  folded: boolean;
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

export function UnfoldDiagram({ folded }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RNN cell folded vs unfolded through time">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {folded ? "RNN Cell 循环表示" : "按时间展开 — 4 个时间步共享同一组权重"}
      </text>

      {folded ? (
        <g>
          {/* single cell with self-loop */}
          <Box x={300} y={130} w={100} h={60} fill="#fce7f3" stroke="#ec4899" label="RNN Cell" sub="(W_x, W_h)" />
          <path d="M 400 140 C 460 100, 460 190, 400 180" fill="none" stroke="#f59e0b" strokeWidth={1.8} markerEnd="url(#loop-arr)" />
          <defs>
            <marker id="loop-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
            </marker>
          </defs>
          <text x={470} y={145} fontSize={10} fill="#92400e">h_t-1</text>

          <Arrow x1={220} y1={160} x2={295} y2={160} color="#3b82f6" id="fold-in" />
          <Box x={140} y={140} w={70} h={40} fill="#dbeafe" stroke="#3b82f6" label="x_t" />

          <Arrow x1={405} y1={130} x2={405} y2={90} color="#10b981" id="fold-out" />
          <Box x={355} y={50} w={100} h={40} fill="#ecfdf5" stroke="#10b981" label="y_t" />
        </g>
      ) : (
        <g>
          {[0, 1, 2, 3].map((t) => {
            const x = 60 + t * 160;
            return (
              <g key={t}>
                <Box x={x} y={80} w={80} h={40} fill="#dbeafe" stroke="#3b82f6" label={`x_${t + 1}`} />
                <Arrow x1={x + 40} y1={120} x2={x + 40} y2={150} color="#3b82f6" id={`u-in-${t}`} />
                <Box x={x} y={150} w={80} h={50} fill="#fce7f3" stroke="#ec4899" label={`h_${t + 1}`} />
                <Arrow x1={x + 40} y1={200} x2={x + 40} y2={230} color="#10b981" id={`u-out-${t}`} />
                <Box x={x} y={230} w={80} h={36} fill="#ecfdf5" stroke="#10b981" label={`y_${t + 1}`} />
                {t < 3 && (
                  <Arrow x1={x + 80} y1={175} x2={x + 160} y2={175} color="#f59e0b" id={`u-h-${t}`} />
                )}
              </g>
            );
          })}
          <text x={W / 2} y={295} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#92400e">
            黄色箭头 = h_t-1 → h_t,所有 cell 用同一组 (W_x, W_h, W_y)
          </text>
        </g>
      )}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        h_t = tanh(W_x·x_t + W_h·h_{"{t-1}"} + b) · 展开后等价深度 T 前馈网络
      </text>
    </svg>
  );
}
