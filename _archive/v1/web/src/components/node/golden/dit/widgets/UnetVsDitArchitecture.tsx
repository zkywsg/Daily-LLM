const W = 700;
const H = 340;

interface Props {
  side: "unet" | "dit" | "both";
}

function Box({ x, y, w, h, fill, stroke, label, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 + 3} textAnchor="middle" fontSize={10} fontWeight={600} fill="#1f2937">{label}</text>
    </g>
  );
}

export function UnetVsDitArchitecture({ side }: Props) {
  const dimU = side === "dit" ? 0.4 : 1;
  const dimD = side === "unet" ? 0.4 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="U-Net vs DiT backbone">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        U-Net (LDM/ADM) vs DiT — diffusion backbone 范式对比
      </text>

      {/* === U-Net 上 === */}
      <g opacity={dimU}>
        <text x={W / 2} y={50} textAnchor="middle" fontSize={12} fontWeight={700} fill="#831843">
          U-Net (LDM) · encoder-decoder + skip + 局部卷积
        </text>
        {/* Encoder 漏斗下降 */}
        <Box x={40}  y={70} w={70} h={32} fill="#dbeafe" stroke="#3b82f6" label="latent" />
        <Box x={130} y={75} w={60} h={26} fill="#fce7f3" stroke="#ec4899" label="ConvBlk" />
        <Box x={210} y={80} w={50} h={20} fill="#fce7f3" stroke="#ec4899" label="↓ 2×" />
        <Box x={280} y={85} w={50} h={16} fill="#fce7f3" stroke="#ec4899" label="↓ 2×" />
        <Box x={350} y={88} w={50} h={12} fill="#fef3c7" stroke="#f59e0b" label="mid" />
        {/* Decoder 漏斗上升 */}
        <Box x={420} y={85} w={50} h={16} fill="#fce7f3" stroke="#ec4899" label="↑ 2×" />
        <Box x={490} y={80} w={50} h={20} fill="#fce7f3" stroke="#ec4899" label="↑ 2×" />
        <Box x={560} y={75} w={60} h={26} fill="#fce7f3" stroke="#ec4899" label="ConvBlk" />
        <Box x={640} y={70} w={45} h={32} fill="#dbeafe" stroke="#3b82f6" label="out" />

        {/* skip connections */}
        <path d="M 160 70 Q 365 30 590 70" fill="none" stroke="#10b981" strokeWidth={1.4} strokeDasharray="4 3" />
        <path d="M 235 75 Q 365 45 515 80" fill="none" stroke="#10b981" strokeWidth={1.4} strokeDasharray="4 3" />
        <text x={365} y={28} textAnchor="middle" fontSize={9} fill="#10b981">skip connections</text>

        <text x={W / 2} y={130} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
          归纳偏置:local conv + multi-scale · 优势在小数据 · scaling 不干净
        </text>
      </g>

      {/* 分隔线 */}
      <line x1={20} y1={160} x2={W - 20} y2={160} stroke="#e5e7eb" />

      {/* === DiT 下 === */}
      <g opacity={dimD}>
        <text x={W / 2} y={186} textAnchor="middle" fontSize={12} fontWeight={700} fill="#065f46">
          DiT · patchify → Transformer × N → un-patchify
        </text>

        <Box x={40}  y={210} w={70} h={32} fill="#dbeafe" stroke="#3b82f6" label="latent" />
        <Box x={130} y={210} w={70} h={32} fill="#fef3c7" stroke="#f59e0b" label="patchify" />
        {/* tokens */}
        {[0, 1, 2, 3, 4].map((i) => (
          <rect key={i} x={220 + i * 11} y={216} width={9} height={20} fill="#fce7f3" stroke="#ec4899" strokeWidth={0.8} rx={1.5} />
        ))}
        <text x={285} y={228} fontSize={9} fill="#9ca3af">…</text>
        <text x={250} y={252} textAnchor="middle" fontSize={9} fill="#6b7280">T 个 token</text>

        {/* N stacked DiT blocks */}
        <Box x={310} y={210} w={170} h={32} fill="#fce7f3" stroke="#ec4899" label="DiT Block × 28" />
        <text x={395} y={252} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#831843">每 block:adaLN-Zero + attn + ffn</text>

        <Box x={490} y={210} w={80} h={32} fill="#fef3c7" stroke="#f59e0b" label="un-patchify" />
        <Box x={580} y={210} w={70} h={32} fill="#dbeafe" stroke="#3b82f6" label="latent" />

        {/* 箭头链 */}
        <defs>
          <marker id="dit-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
          </marker>
        </defs>
        {[[110, 226, 130, 226], [200, 226, 220, 226], [296, 226, 310, 226], [480, 226, 490, 226], [570, 226, 580, 226]].map((c, i) => (
          <line key={i} x1={c[0]} y1={c[1]} x2={c[2]} y2={c[3]} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#dit-arr)" />
        ))}

        <text x={W / 2} y={280} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
          无归纳偏置 · 数据自己学 patch 间 attention · scaling 干净幂律
        </text>
      </g>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        DiT 把视觉生成纳入 Transformer 统一框架 — 直接催生 Sora / SD3 / FLUX
      </text>
    </svg>
  );
}
