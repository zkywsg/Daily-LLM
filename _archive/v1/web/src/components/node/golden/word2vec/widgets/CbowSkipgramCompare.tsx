const W = 700;
const H = 360;

// 左: CBOW — 周围 context → 中心词 fox
// 右: Skip-gram — 中心词 fox → 周围 context

interface Props {
  highlightSide?: "cbow" | "skipgram" | "both";
}

function Word({ x, y, w, h, label, fill, stroke }: {
  x: number; y: number; w: number; h: number; label: string; fill: string; stroke: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.2} />
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
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
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.3} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function CbowSkipgramCompare({ highlightSide = "both" }: Props) {
  const dimL = highlightSide === "skipgram" ? 0.35 : 1;
  const dimR = highlightSide === "cbow" ? 0.35 : 1;

  // CBOW 左半区:context 在四角 → projection (中) → center (右)
  const CTX_L = [
    { label: "the",   x: 20,  y: 40 },
    { label: "quick", x: 20,  y: 100 },
    { label: "brown", x: 20,  y: 200 },
    { label: "jumps", x: 20,  y: 260 },
  ];
  const PROJ_L = { x: 140, y: 160 };
  const CENTER_L = { x: 250, y: 160, label: "fox" };

  // Skip-gram 右半区:center (左) → projection (中) → context (四个)
  const CENTER_R = { x: 380, y: 160, label: "fox" };
  const PROJ_R = { x: 490, y: 160 };
  const CTX_R = [
    { label: "the",   x: 600, y: 40 },
    { label: "quick", x: 600, y: 100 },
    { label: "brown", x: 600, y: 200 },
    { label: "jumps", x: 600, y: 260 },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CBOW vs Skip-gram side-by-side comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        CBOW 与 Skip-gram — 互为对偶的两种任务
      </text>

      {/* 中线 */}
      <line x1={W / 2} y1={36} x2={W / 2} y2={H - 24} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4 4" />

      {/* === CBOW 左 === */}
      <g opacity={dimL}>
        <text x={150} y={36} textAnchor="middle" fontSize={12} fontWeight={700} fill="#92400e">CBOW · context → center</text>
        {CTX_L.map((c, i) => (
          <Word key={i} x={c.x} y={c.y} w={70} h={28} label={c.label} fill="#fef3c7" stroke="#f59e0b" />
        ))}
        {/* projection node */}
        <circle cx={PROJ_L.x} cy={PROJ_L.y} r={22} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
        <text x={PROJ_L.x} y={PROJ_L.y + 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="#831843">SUM/AVG</text>

        {/* arrows context → proj */}
        {CTX_L.map((c, i) => (
          <Arrow key={i} id={`cbow-arr-${i}`} x1={c.x + 70} y1={c.y + 14} x2={PROJ_L.x - 22} y2={PROJ_L.y} color="#f59e0b" />
        ))}

        {/* proj → center */}
        <Arrow id="cbow-pc" x1={PROJ_L.x + 22} y1={PROJ_L.y} x2={CENTER_L.x - 28} y2={CENTER_L.y} color="#ec4899" />
        <Word x={CENTER_L.x - 28} y={CENTER_L.y - 14} w={56} h={28} label={CENTER_L.label} fill="#ecfdf5" stroke="#10b981" />

        <text x={150} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
          看 2k 个 context 算一次梯度 → 大数据更快
        </text>
      </g>

      {/* === Skip-gram 右 === */}
      <g opacity={dimR}>
        <text x={W - 150} y={36} textAnchor="middle" fontSize={12} fontWeight={700} fill="#065f46">Skip-gram · center → context</text>
        <Word x={CENTER_R.x - 28} y={CENTER_R.y - 14} w={56} h={28} label={CENTER_R.label} fill="#ecfdf5" stroke="#10b981" />

        {/* center → proj */}
        <Arrow id="sg-cp" x1={CENTER_R.x + 28} y1={CENTER_R.y} x2={PROJ_R.x - 22} y2={PROJ_R.y} color="#10b981" />
        <circle cx={PROJ_R.x} cy={PROJ_R.y} r={22} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
        <text x={PROJ_R.x} y={PROJ_R.y + 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="#831843">proj</text>

        {/* proj → context */}
        {CTX_R.map((c, i) => (
          <Arrow key={i} id={`sg-arr-${i}`} x1={PROJ_R.x + 22} y1={PROJ_R.y} x2={c.x} y2={c.y + 14} color="#ec4899" />
        ))}
        {CTX_R.map((c, i) => (
          <Word key={i} x={c.x} y={c.y} w={70} h={28} label={c.label} fill="#fef3c7" stroke="#f59e0b" />
        ))}

        <text x={W - 150} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
          每个低频词被预测 2k 次 → 罕见词更准
        </text>
      </g>
    </svg>
  );
}
