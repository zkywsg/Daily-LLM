const W = 700;
const H = 360;

interface Props {
  highlight: "self" | "cross" | "ffn";
}

// U-Net block 内部:Self-Attn → Cross-Attn → FFN
// Cross-Attn 里 Q ← latent, K/V ← condition
function Box({ x, y, w, h, fill, stroke, label, sub, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={700} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>}
    </g>
  );
}

export function UnetBlockCrossAttn({ highlight }: Props) {
  const dimSelf = highlight === "self" ? 1 : 0.4;
  const dimCross = highlight === "cross" ? 1 : 0.4;
  const dimFFN = highlight === "ffn" ? 1 : 0.4;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="U-Net block self-attn cross-attn FFN">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        U-Net block 内部 — Self-Attn → Cross-Attn → FFN
      </text>

      {/* latent input */}
      <Box x={20} y={150} w={100} h={50} fill="#dbeafe" stroke="#3b82f6" label="latent z_t" sub="64²×4" />

      {/* self-attn */}
      <Box x={170} y={140} w={120} h={70} fill="#fce7f3" stroke="#ec4899" label="Self-Attn" sub="z ↔ z" opacity={dimSelf} />
      <line x1={120} y1={175} x2={170} y2={175} stroke="#9ca3af" markerEnd="url(#u-arr)" strokeWidth={1.4} />

      {/* cross-attn */}
      <Box x={320} y={140} w={140} h={70} fill="#fef3c7" stroke="#f59e0b" label="Cross-Attn" sub="Q←z, K/V←c" opacity={dimCross} />
      <line x1={290} y1={175} x2={320} y2={175} stroke="#9ca3af" markerEnd="url(#u-arr)" strokeWidth={1.4} />

      {/* FFN */}
      <Box x={490} y={140} w={110} h={70} fill="#ecfdf5" stroke="#10b981" label="FFN" sub="MLP × 4" opacity={dimFFN} />
      <line x1={460} y1={175} x2={490} y2={175} stroke="#9ca3af" markerEnd="url(#u-arr)" strokeWidth={1.4} />

      {/* output back */}
      <Box x={620} y={150} w={60} h={50} fill="#dbeafe" stroke="#3b82f6" label="z'" />
      <line x1={600} y1={175} x2={620} y2={175} stroke="#9ca3af" markerEnd="url(#u-arr)" strokeWidth={1.4} />

      <defs>
        <marker id="u-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* condition encoders */}
      <text x={W / 2} y={250} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">
        condition c — 三种 encoder 全部输出 K/V 序列,接同一个 cross-attention
      </text>

      {[
        { x: 70,  label: "prompt 文本", sub: "CLIP text", shape: "[77, 768]", color: "#fef3c7", stroke: "#f59e0b" },
        { x: 290, label: "类别 / 标签", sub: "embedding", shape: "[1, 768]",  color: "#dbeafe", stroke: "#3b82f6" },
        { x: 510, label: "语义图 / 深度", sub: "conv enc", shape: "[H'W', 768]", color: "#fce7f3", stroke: "#ec4899" },
      ].map((cond, i) => (
        <g key={i}>
          <rect x={cond.x} y={270} width={140} height={60} rx={4} fill={cond.color} stroke={cond.stroke} strokeWidth={1.4} opacity={dimCross} />
          <text x={cond.x + 70} y={290} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1f2937">{cond.label}</text>
          <text x={cond.x + 70} y={306} textAnchor="middle" fontSize={9} fill="#6b7280">{cond.sub}</text>
          <text x={cond.x + 70} y={320} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#6b7280">{cond.shape}</text>
          {/* arrow up to cross-attn */}
          <line x1={cond.x + 70} y1={270} x2={390} y2={210} stroke="#f59e0b" strokeWidth={1.2} strokeDasharray="3 3" opacity={dimCross} />
        </g>
      ))}
    </svg>
  );
}
