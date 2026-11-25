const W = 700;
const H = 320;

interface Props {
  highlight: "cond" | "uncond" | "diff" | "scale" | null;
}

function Box({ x, y, w, h, fill, stroke, label, sub, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
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

export function CfgDataflow({ highlight }: Props) {
  const dimC = highlight && highlight !== "cond" ? 0.35 : 1;
  const dimU = highlight && highlight !== "uncond" ? 0.35 : 1;
  const dimD = highlight && highlight !== "diff" ? 0.35 : 1;
  const dimS = highlight && highlight !== "scale" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Classifier-free guidance dataflow">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Classifier-Free Guidance — 减法 + 放大控制条件强度
      </text>

      <Box x={30} y={140} w={90} h={40} fill="#dbeafe" stroke="#3b82f6" label="x_t" />

      <g opacity={dimC}>
        <Arrow x1={120} y1={150} x2={180} y2={100} color="#3b82f6" id="c-a1" />
        <Box x={180} y={70} w={170} h={44} fill="#fce7f3" stroke="#ec4899" label="ε_θ(x_t, t, c)" sub="条件预测" />
      </g>

      <g opacity={dimU}>
        <Arrow x1={120} y1={160} x2={180} y2={200} color="#3b82f6" id="c-a2" />
        <Box x={180} y={180} w={170} h={44} fill="#f3f4f6" stroke="#9ca3af" label="ε_θ(x_t, t, ∅)" sub="无条件预测" />
      </g>

      <g opacity={dimD}>
        <Arrow x1={350} y1={92} x2={400} y2={140} color="#ec4899" id="c-a3" />
        <Arrow x1={350} y1={202} x2={400} y2={150} color="#9ca3af" id="c-a4" />
        <Box x={400} y={125} w={150} h={40} fill="#fef3c7" stroke="#f59e0b" label="ε(c) − ε(∅)" sub="条件特有方向" />
      </g>

      <g opacity={dimS}>
        <Arrow x1={550} y1={145} x2={590} y2={145} color="#f59e0b" id="c-a5" />
        <Box x={590} y={125} w={80} h={40} fill="#fef3c7" stroke="#f59e0b" label="× w" sub="放大" />
      </g>

      {/* final add */}
      <circle cx={310} cy={250} r={18} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.8} />
      <text x={310} y={255} textAnchor="middle" fontSize={14} fontWeight={700} fill="#065f46">+</text>
      <Arrow x1={265} y1={202} x2={294} y2={244} color="#9ca3af" id="c-a6" />
      <Arrow x1={630} y1={165} x2={330} y2={244} color="#f59e0b" id="c-a7" />

      <Box x={380} y={235} w={180} h={40} fill="#ecfdf5" stroke="#10b981" label="ε̃(x_t, t, c)" sub="guided 预测" />
      <Arrow x1={328} y1={250} x2={380} y2={255} color="#10b981" id="c-a8" />

      <rect x={80} y={288} width={W - 160} height={26} rx={4} fill="#fef3c7" fillOpacity={0.5} stroke="#f59e0b" strokeWidth={1} />
      <text x={W / 2} y={305} textAnchor="middle" fontSize={11} fontFamily="ui-monospace, monospace" fill="#92400e">
        ε̃ = ε(∅) + w·[ε(c) − ε(∅)]
      </text>
    </svg>
  );
}
