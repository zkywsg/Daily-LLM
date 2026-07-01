const W = 700;
const H = 380;

interface Props {
  side: "trad" | "style" | "both";
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

function Arrow({ x1, y1, x2, y2, color, id, dashed }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string; dashed?: boolean }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} strokeDasharray={dashed ? "4 3" : undefined} />
    </g>
  );
}

export function TraditionalVsStyleGAN({ side }: Props) {
  const dimT = side === "style" ? 0.35 : 1;
  const dimS = side === "trad" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Traditional GAN vs StyleGAN architecture">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        传统 GAN vs StyleGAN — 架构根本差异
      </text>

      {/* === Traditional GAN === */}
      <g opacity={dimT}>
        <text x={140} y={50} textAnchor="middle" fontSize={12} fontWeight={700} fill="#831843">传统 GAN</text>
        <Box x={80} y={70} w={120} h={36} fill="#fce7f3" stroke="#ec4899" label="z ~ N(0,I)" sub="Gaussian 512" />
        <Arrow x1={140} y1={106} x2={140} y2={130} color="#ec4899" id="t1" />
        <Box x={80} y={130} w={120} h={36} fill="#dbeafe" stroke="#3b82f6" label="Generator" sub="Conv 栈" />
        <Arrow x1={140} y1={166} x2={140} y2={190} color="#3b82f6" id="t2" />
        <Box x={80} y={190} w={120} h={36} fill="#ecfdf5" stroke="#10b981" label="image" sub="1024²" />

        <text x={140} y={260} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#831843">
          z 同时控制姿态 + 肤色 + 发型
        </text>
        <text x={140} y={278} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#831843">
          latent 高度纠缠
        </text>
        <text x={140} y={296} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#9ca3af">
          style mixing 不可行
        </text>
      </g>

      {/* === StyleGAN === */}
      <g opacity={dimS}>
        <text x={500} y={50} textAnchor="middle" fontSize={12} fontWeight={700} fill="#065f46">StyleGAN</text>

        {/* z → mapping → w */}
        <Box x={300} y={70} w={100} h={32} fill="#fce7f3" stroke="#ec4899" label="z ~ N(0,I)" />
        <Arrow x1={350} y1={102} x2={350} y2={120} color="#ec4899" id="s1" />
        <Box x={300} y={120} w={100} h={32} fill="#fef3c7" stroke="#f59e0b" label="8 层 MLP" sub="mapping" />
        <Arrow x1={350} y1={152} x2={350} y2={170} color="#f59e0b" id="s2" />
        <Box x={300} y={170} w={100} h={32} fill="#fef3c7" stroke="#f59e0b" label="w ∈ W" sub="解纠缠" />

        {/* Constant → G blocks */}
        <Box x={460} y={70} w={120} h={36} fill="#dbeafe" stroke="#3b82f6" label="学到的常量" sub="4×4×512" />
        <Arrow x1={520} y1={106} x2={520} y2={125} color="#3b82f6" id="s3" />

        {/* G blocks with AdaIN */}
        {[0, 1, 2, 3].map((i) => (
          <g key={i}>
            <Box x={460} y={125 + i * 40} w={120} h={32}
                 fill="#ecfdf5" stroke="#10b981"
                 label={`Block ${i + 1} + AdaIN`}
                 sub={i === 0 ? "4×4" : i === 1 ? "8×8" : i === 2 ? "32×32" : "1024×1024"} />
            {i < 3 && (
              <Arrow x1={520} y1={157 + i * 40} x2={520} y2={165 + i * 40} color="#10b981" id={`s-blk-${i}`} />
            )}
            {/* w 注入箭头 */}
            <Arrow x1={400} y1={186} x2={460} y2={141 + i * 40} color="#f59e0b" id={`s-w-${i}`} dashed />
          </g>
        ))}

        <text x={620} y={186} fontSize={10} fontStyle="italic" fill="#92400e">w 注入每层</text>

        <Arrow x1={520} y1={285} x2={520} y2={310} color="#10b981" id="s-out" />
        <Box x={460} y={310} w={120} h={32} fill="#ecfdf5" stroke="#10b981" label="image 1024²" />

        <text x={520} y={360} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#065f46">
          z → w 解纠缠 · w 注入每层 · style mixing 可行
        </text>
      </g>

      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        G 输入是学到的常量,不是 z · z 只决定 w · w 通过 AdaIN 注入每层
      </text>
    </svg>
  );
}
