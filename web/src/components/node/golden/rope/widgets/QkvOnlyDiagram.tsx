const W = 700;
const H = 260;

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

export function QkvOnlyDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RoPE only applied to Q and K">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        只对 Q/K 应用 RoPE — V 承载内容,不该被位置污染
      </text>

      <Box x={40} y={60} w={90} h={40} fill="#fce7f3" stroke="#ec4899" label="Q" />
      <Arrow x1={130} y1={80} x2={170} y2={80} color="#ec4899" id="q-a1" />
      <Box x={170} y={60} w={110} h={40} fill="#fce7f3" stroke="#ec4899" label="RoPE(Q)" sub="旋转" />

      <Box x={40} y={130} w={90} h={40} fill="#dbeafe" stroke="#3b82f6" label="K" />
      <Arrow x1={130} y1={150} x2={170} y2={150} color="#3b82f6" id="k-a1" />
      <Box x={170} y={130} w={110} h={40} fill="#dbeafe" stroke="#3b82f6" label="RoPE(K)" sub="旋转" />

      <Box x={40} y={200} w={90} h={40} fill="#f3f4f6" stroke="#9ca3af" label="V" />
      <text x={170} y={225} fontSize={11} fontStyle="italic" fill="#9ca3af">← 不动,直接送入 attention</text>

      <Arrow x1={280} y1={80} x2={340} y2={110} color="#ec4899" id="q-a2" />
      <Arrow x1={280} y1={150} x2={340} y2={110} color="#3b82f6" id="k-a2" />
      <Box x={340} y={90} w={120} h={40} fill="#fef3c7" stroke="#f59e0b" label="QKᵀ/√d" sub="scores" />

      <Arrow x1={460} y1={110} x2={510} y2={110} color="#f59e0b" id="s-a" />
      <Box x={510} y={90} w={100} h={40} fill="#fef3c7" stroke="#f59e0b" label="softmax" />

      <Arrow x1={130} y1={220} x2={560} y2={150} color="#9ca3af" id="v-a" />
      <Arrow x1={560} y1={130} x2={560} y2={165} color="#10b981" id="out-a" />
      <Box x={500} y={165} w={130} h={40} fill="#ecfdf5" stroke="#10b981" label="attn·V" sub="输出" />

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        cos/sin 表只依赖位置和频率,一次预计算,推理零额外开销
      </text>
    </svg>
  );
}
