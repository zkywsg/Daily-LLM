const W = 700;
const H = 320;

interface Props {
  highlight: "vision" | "perceiver" | "llm" | "xattn" | null;
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

export function FlamingoArchitecture({ highlight }: Props) {
  const dimV = highlight && highlight !== "vision" ? 0.35 : 1;
  const dimP = highlight && highlight !== "perceiver" ? 0.35 : 1;
  const dimL = highlight && highlight !== "llm" ? 0.35 : 1;
  const dimX = highlight && highlight !== "xattn" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Flamingo architecture">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Flamingo 架构 — 冻结视觉 + 冻结 LLM + 可训练桥接
      </text>

      {/* vision */}
      <g opacity={dimV}>
        <Box x={30} y={60} w={120} h={40} fill="#dbeafe" stroke="#3b82f6" label="Image/Video" />
        <Arrow x1={90} y1={100} x2={90} y2={130} color="#3b82f6" id="fa-a1" />
        <Box x={30} y={130} w={120} h={40} fill="#f3f4f6" stroke="#9ca3af" label="NF-ResNet F6" sub="冻结 435M" />
      </g>

      {/* perceiver */}
      <g opacity={dimP}>
        <Arrow x1={90} y1={170} x2={90} y2={200} color="#9ca3af" id="fa-a2" />
        <Box x={30} y={200} w={120} h={44} fill="#fce7f3" stroke="#ec4899" label="Perceiver Resampler" sub="可训练 200M" />
        <text x={90} y={260} textAnchor="middle" fontSize={9} fill="#831843">→ 64 visual tokens(固定)</text>
      </g>

      {/* LLM */}
      <g opacity={dimL}>
        <Box x={250} y={60} w={130} h={40} fill="#dbeafe" stroke="#3b82f6" label="Text tokens" />
        <Arrow x1={315} y1={100} x2={315} y2={130} color="#3b82f6" id="fa-a3" />
        <Box x={230} y={130} w={170} h={110} fill="#f3f4f6" stroke="#9ca3af" label="Chinchilla 70B" sub="冻结 · self-attn + FFN" />
      </g>

      {/* cross attn */}
      <g opacity={dimX}>
        <Arrow x1={150} y1={222} x2={230} y2={185} color="#ec4899" id="fa-a4" />
        <Box x={470} y={150} w={170} h={70} fill="#fef3c7" stroke="#f59e0b" label="Gated Cross-Attn" sub="每 7 层插入 · 可训练 200M" />
        <Arrow x1={400} y1={185} x2={470} y2={185} color="#f59e0b" id="fa-a5" />
        <text x={555} y={240} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#92400e">tanh(0)=0 初始 identity</text>
      </g>

      <Arrow x1={555} y1={150} x2={555} y2={110} color="#10b981" id="fa-a6" />
      <Box x={470} y={70} w={170} h={40} fill="#ecfdf5" stroke="#10b981" label="Generated text" />

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        可训练参数只占 10B / 80B ≈ 12.5% · 冻结部分保留 LLM 全部语言能力
      </text>
    </svg>
  );
}
