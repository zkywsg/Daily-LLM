const W = 700;
const H = 260;

interface Props {
  side: "cascade" | "ldm" | "both";
}

function Box({ x, y, w, h, fill, stroke, label, sub }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={10} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

export function CascadeVsLdm({ side }: Props) {
  const dimC = side === "ldm" ? 0.35 : 1;
  const dimL = side === "cascade" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Cascade vs LDM approach">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Imagen(Cascade)vs LDM(Latent)— 两条工业路线
      </text>

      <g opacity={dimC}>
        <text x={175} y={50} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">Imagen · Cascade(pixel 空间)</text>
        <Box x={40} y={65} w={90} h={36} fill="#fce7f3" stroke="#ec4899" label="64×64" />
        <Box x={140} y={65} w={90} h={36} fill="#fce7f3" stroke="#ec4899" label="256×256" />
        <Box x={240} y={65} w={90} h={36} fill="#fce7f3" stroke="#ec4899" label="1024×1024" />
        <text x={175} y={125} textAnchor="middle" fontSize={9} fill="#374151">3 次 diffusion · 无 VAE 损失</text>
        <text x={175} y={140} textAnchor="middle" fontSize={9} fill="#9ca3af">Google 系:Imagen/Imagen Video/Lumiere</text>
      </g>

      <line x1={360} y1={40} x2={360} y2={160} stroke="#e5e7eb" strokeDasharray="3 3" />

      <g opacity={dimL} transform="translate(380, 0)">
        <text x={155} y={50} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">LDM · Latent(压缩空间)</text>
        <Box x={30} y={65} w={90} h={36} fill="#dbeafe" stroke="#3b82f6" label="VAE encode" sub="→ 64×64×4" />
        <Box x={130} y={65} w={90} h={36} fill="#ecfdf5" stroke="#10b981" label="1× diffusion" sub="latent 空间" />
        <Box x={230} y={65} w={90} h={36} fill="#dbeafe" stroke="#3b82f6" label="VAE decode" sub="→ pixel" />
        <text x={155} y={125} textAnchor="middle" fontSize={9} fill="#374151">1 次 diffusion · 算力省 64×</text>
        <text x={155} y={140} textAnchor="middle" fontSize={9} fill="#9ca3af">社区系:SD/Midjourney/Flux</text>
      </g>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        Cascade 质量可能略好但工程复杂 3×;LDM 塞进消费 GPU — 两条路并存至今
      </text>
    </svg>
  );
}
