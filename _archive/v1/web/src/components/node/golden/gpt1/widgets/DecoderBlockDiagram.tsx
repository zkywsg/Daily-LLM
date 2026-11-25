const W = 700;
const H = 340;

interface Props {
  highlight: "attn" | "ffn" | "gelu" | "tying" | null;
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

export function DecoderBlockDiagram({ highlight }: Props) {
  const dimAttn = highlight && highlight !== "attn" ? 0.35 : 1;
  const dimFfn = highlight && highlight !== "ffn" ? 0.35 : 1;
  const dimGelu = highlight && highlight !== "gelu" ? 0.35 : 1;
  const dimTying = highlight && highlight !== "tying" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-1 decoder block diagram">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        GPT-1 Decoder Block × 12 — Post-LN + Masked Attn + GELU FFN
      </text>

      {/* Input tokens */}
      <Box x={30} y={280} w={100} h={30} fill="#dbeafe" stroke="#3b82f6" label="tok_emb + pos_emb" />
      <Arrow x1={80} y1={280} x2={80} y2={250} color="#3b82f6" id="in-a" />

      {/* Masked Attn */}
      <g opacity={dimAttn}>
        <Box x={30} y={210} w={140} h={40} fill="#fce7f3" stroke="#ec4899"
             label="Masked Self-Attn" sub="causal mask · 只看 [0,t]" />
      </g>
      <Arrow x1={100} y1={210} x2={100} y2={190} color="#ec4899" id="attn-a" />
      <Box x={30} y={160} w={140} h={30} fill="#fef3c7" stroke="#f59e0b" label="Add + LayerNorm" sub="Post-LN" />
      <Arrow x1={100} y1={160} x2={100} y2={135} color="#f59e0b" id="ln1-a" />

      {/* FFN with GELU */}
      <g opacity={dimFfn}>
        <Box x={30} y={95} w={140} h={40} fill="#ecfdf5" stroke="#10b981" label="FFN (d_ff=3072)" sub="Linear → GELU → Linear" />
      </g>
      <g opacity={dimGelu}>
        <circle cx={100} cy={115} r={3} fill="none" />
        <text x={190} y={110} fontSize={10} fontWeight={700} fill="#065f46">← GELU 不是 ReLU</text>
      </g>
      <Arrow x1={100} y1={95} x2={100} y2={75} color="#10b981" id="ffn-a" />
      <Box x={30} y={45} w={140} h={30} fill="#fef3c7" stroke="#f59e0b" label="Add + LayerNorm" />

      <text x={100} y={330} textAnchor="middle" fontSize={10} fill="#9ca3af">× 12 层</text>

      {/* 右侧:LM head + weight tying */}
      <g opacity={dimTying}>
        <Box x={350} y={95} w={150} h={40} fill="#dbeafe" stroke="#3b82f6" label="tok_emb" sub="[vocab, d_model]" />
        <Box x={350} y={45} w={150} h={40} fill="#dbeafe" stroke="#3b82f6" label="lm_head (tied)" sub="weight = tok_emb.weight" />

        <line x1={430} y1={95} x2={430} y2={85} stroke="#3b82f6" strokeWidth={2} strokeDasharray="4 3" />
        <text x={520} y={70} fontSize={10} fontWeight={700} fill="#1e40af">weight tying</text>
        <text x={520} y={86} fontSize={9} fill="#6b7280">省 30%+ 参数</text>

        <Box x={550} y={130} w={130} h={40} fill="#fce7f3" stroke="#ec4899" label="logits" sub="[B,T,vocab]" />
        <Arrow x1={500} y1={150} x2={550} y2={150} color="#3b82f6" id="head-a" />
      </g>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        12 层 · d_model=768 · h=12 · d_ff=3072 · 117M 参数
      </text>
    </svg>
  );
}
