const W = 700;
const H = 320;

interface Props {
  side: "post" | "pre" | "both";
}

// 左 Post-LN: x → attn → add → ln → ffn → add → ln
// 右 Pre-LN:  x → ln → attn → add → ln → ffn → add

function Block({ x, y, w, h, fill, stroke, label }: { x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
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
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function PreVsPostLN({ side }: Props) {
  const dimPost = side === "pre" ? 0.35 : 1;
  const dimPre = side === "post" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Post-LN vs Pre-LN comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Post-LN(GPT-1)vs Pre-LN(GPT-2)— 48 层稳定训练的关键
      </text>

      {/* 中线分隔 */}
      <line x1={W / 2} y1={42} x2={W / 2} y2={H - 24} stroke="#e5e7eb" strokeDasharray="3 3" />

      {/* === POST-LN (left) === */}
      <g opacity={dimPost}>
        <text x={W / 4} y={42} textAnchor="middle" fontSize={12} fontWeight={700} fill="#ec4899">Post-LN: x = LN(x + Sublayer(x))</text>
        {/* input */}
        <Block x={30} y={60} w={70} h={30} fill="#dbeafe" stroke="#3b82f6" label="x" />
        {/* attn */}
        <Block x={130} y={60} w={80} h={30} fill="#fce7f3" stroke="#ec4899" label="Attn" />
        <Arrow x1={100} y1={75} x2={130} y2={75} color="#9ca3af" id="post1" />
        {/* + */}
        <circle cx={230} cy={75} r={10} fill="#fff" stroke="#9ca3af" />
        <text x={230} y={79} textAnchor="middle" fontSize={12}>+</text>
        <Arrow x1={210} y1={75} x2={220} y2={75} color="#9ca3af" id="post2" />
        {/* residual */}
        <line x1={65} y1={60} x2={65} y2={120} stroke="#9ca3af" strokeDasharray="3 3" />
        <line x1={65} y1={120} x2={230} y2={120} stroke="#9ca3af" strokeDasharray="3 3" />
        <line x1={230} y1={120} x2={230} y2={85} stroke="#9ca3af" strokeDasharray="3 3" />
        {/* ln after */}
        <Block x={258} y={60} w={50} h={30} fill="#fef3c7" stroke="#f59e0b" label="LN" />
        <Arrow x1={240} y1={75} x2={258} y2={75} color="#9ca3af" id="post3" />

        {/* stability note */}
        <text x={W / 4} y={170} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">深度 ≥ 24 层时</text>
        <text x={W / 4} y={186} textAnchor="middle" fontSize={11} fill="#831843">梯度方差爆炸 / 训练发散</text>
        <text x={W / 4} y={202} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#9ca3af">需要精细 warmup 才不崩</text>

        {/* gradient indicator */}
        <text x={W / 4} y={240} textAnchor="middle" fontSize={11} fontWeight={600} fill="#374151">48 层反传梯度方差</text>
        <rect x={W / 4 - 90} y={250} width={180} height={20} fill="#fce7f3" stroke="#ec4899" />
        <text x={W / 4} y={264} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">~ 100×(不稳定)</text>
      </g>

      {/* === PRE-LN (right) === */}
      <g opacity={dimPre} transform={`translate(${W / 2}, 0)`}>
        <text x={W / 4} y={42} textAnchor="middle" fontSize={12} fontWeight={700} fill="#10b981">Pre-LN: x = x + Sublayer(LN(x))</text>
        {/* input */}
        <Block x={30} y={60} w={50} h={30} fill="#dbeafe" stroke="#3b82f6" label="x" />
        {/* ln first */}
        <Block x={100} y={60} w={50} h={30} fill="#fef3c7" stroke="#f59e0b" label="LN" />
        <Arrow x1={80} y1={75} x2={100} y2={75} color="#9ca3af" id="pre1" />
        {/* attn */}
        <Block x={170} y={60} w={70} h={30} fill="#fce7f3" stroke="#ec4899" label="Attn" />
        <Arrow x1={150} y1={75} x2={170} y2={75} color="#9ca3af" id="pre2" />
        {/* + */}
        <circle cx={258} cy={75} r={10} fill="#fff" stroke="#9ca3af" />
        <text x={258} y={79} textAnchor="middle" fontSize={12}>+</text>
        <Arrow x1={240} y1={75} x2={250} y2={75} color="#9ca3af" id="pre3" />
        {/* residual (直接 from x) */}
        <line x1={55} y1={60} x2={55} y2={120} stroke="#10b981" strokeDasharray="3 3" />
        <line x1={55} y1={120} x2={258} y2={120} stroke="#10b981" strokeDasharray="3 3" />
        <line x1={258} y1={120} x2={258} y2={85} stroke="#10b981" strokeDasharray="3 3" />
        <text x={155} y={114} fontSize={9} fontStyle="italic" fill="#10b981">残差是 \"纯通道\"</text>

        {/* stability */}
        <text x={W / 4} y={170} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">48 层从头稳定训</text>
        <text x={W / 4} y={186} textAnchor="middle" fontSize={11} fill="#065f46">梯度方差近似不变</text>
        <text x={W / 4} y={202} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#9ca3af">warmup 仍要,但容错大很多</text>

        {/* gradient indicator */}
        <text x={W / 4} y={240} textAnchor="middle" fontSize={11} fontWeight={600} fill="#374151">48 层反传梯度方差</text>
        <rect x={W / 4 - 90} y={250} width={180} height={20} fill="#ecfdf5" stroke="#10b981" />
        <text x={W / 4} y={264} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">~ 1×(稳定)</text>
      </g>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        Pre-LN 让残差成纯通道 — 反传梯度沿残差直传不被 LN 挤压
      </text>
    </svg>
  );
}
