const W = 700;
const H = 380;

interface Props {
  highlight: "condition" | "norm" | "scale" | "gate" | null;
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

export function AdaLNZeroBlock({ highlight }: Props) {
  const dimC = highlight && highlight !== "condition" ? 0.4 : 1;
  const dimN = highlight && highlight !== "norm" ? 0.4 : 1;
  const dimS = highlight && highlight !== "scale" ? 0.4 : 1;
  const dimG = highlight && highlight !== "gate" ? 0.4 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="adaLN-Zero DiT block">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        adaLN-Zero DiT Block — timestep + class 条件如何注入
      </text>

      {/* === 左侧 condition path === */}
      <g opacity={dimC}>
        <Box x={20}  y={60} w={130} h={36} fill="#fef3c7" stroke="#f59e0b" label="timestep t" />
        <Box x={20}  y={110} w={130} h={36} fill="#fef3c7" stroke="#f59e0b" label="class label c" />
        <text x={85} y={170} textAnchor="middle" fontSize={11} fill="#6b7280">+</text>
        <Box x={20}  y={180} w={130} h={40} fill="#fce7f3" stroke="#ec4899" label="MLP (6 × d 维)" sub="Zero init 最后一层" />

        {/* 6 个 chunk */}
        <text x={20} y={245} fontSize={10} fontWeight={600} fill="#831843">→ chunk(6, dim=-1):</text>
        <text x={20} y={262} fontSize={10} fontFamily="ui-monospace, monospace" fill="#831843">γ₁ β₁ α₁ γ₂ β₂ α₂</text>

        <Arrow x1={150} y1={78} x2={170} y2={195} color="#f59e0b" id="c-a1" />
        <Arrow x1={150} y1={128} x2={170} y2={200} color="#f59e0b" id="c-a2" />
      </g>

      {/* === 右侧 token forward path === */}
      <g>
        {/* 输入 token */}
        <Box x={250} y={60} w={100} h={36} fill="#dbeafe" stroke="#3b82f6" label="x (tokens)" opacity={1} />

        {/* attention sub-block */}
        <g opacity={dimN}>
          <Box x={250} y={120} w={100} h={32} fill="#ecfdf5" stroke="#10b981" label="LayerNorm" />
        </g>
        <g opacity={dimS}>
          <Box x={250} y={160} w={100} h={32} fill="#fef3c7" stroke="#f59e0b" label="× (1+γ₁) + β₁" sub="adaLN scale/shift" />
        </g>
        <Box x={250} y={200} w={100} h={32} fill="#fce7f3" stroke="#ec4899" label="Self-Attn" />
        <g opacity={dimG}>
          <Box x={250} y={240} w={100} h={32} fill="#fef3c7" stroke="#f59e0b" label="× α₁ (gate)" sub="残差缩放" />
        </g>

        {/* residual */}
        <line x1={245} y1={78} x2={235} y2={78} stroke="#9ca3af" strokeWidth={1.2} />
        <line x1={235} y1={78} x2={235} y2={290} stroke="#9ca3af" strokeWidth={1.2} strokeDasharray="3 3" />
        <line x1={235} y1={290} x2={245} y2={290} stroke="#9ca3af" strokeWidth={1.2} />
        <circle cx={300} cy={290} r={10} fill="#fff" stroke="#9ca3af" />
        <text x={300} y={294} textAnchor="middle" fontSize={11}>+</text>

        {/* arrows */}
        {[[300, 96, 300, 120], [300, 152, 300, 160], [300, 192, 300, 200], [300, 232, 300, 240], [300, 272, 300, 280]].map((c, i) => (
          <Arrow key={i} x1={c[0]} y1={c[1]} x2={c[2]} y2={c[3]} color="#9ca3af" id={`fwd-${i}`} />
        ))}

        <text x={400} y={210} fontSize={10} fontWeight={600} fill="#374151">attention sub-block</text>
      </g>

      {/* FFN sub-block (右下区域) */}
      <g>
        <text x={500} y={80} textAnchor="middle" fontSize={11} fontWeight={700} fill="#374151">FFN sub-block</text>
        <Box x={450} y={100} w={100} h={32} fill="#ecfdf5" stroke="#10b981" label="LayerNorm" opacity={dimN} />
        <Box x={450} y={140} w={100} h={32} fill="#fef3c7" stroke="#f59e0b" label="× (1+γ₂) + β₂" opacity={dimS} />
        <Box x={450} y={180} w={100} h={32} fill="#fce7f3" stroke="#ec4899" label="MLP" />
        <Box x={450} y={220} w={100} h={32} fill="#fef3c7" stroke="#f59e0b" label="× α₂ (gate)" opacity={dimG} />
        <text x={500} y={285} textAnchor="middle" fontSize={10} fill="#6b7280">+ 残差 → 下一 block</text>

        <Arrow x1={500} y1={132} x2={500} y2={140} color="#9ca3af" id="ffn-1" />
        <Arrow x1={500} y1={172} x2={500} y2={180} color="#9ca3af" id="ffn-2" />
        <Arrow x1={500} y1={212} x2={500} y2={220} color="#9ca3af" id="ffn-3" />
      </g>

      {/* 标 zero init 关键说明 */}
      <rect x={50} y={310} width={W - 100} height={50} rx={6} fill="#ecfdf5" stroke="#10b981" strokeOpacity={0.5} strokeWidth={1.2} />
      <text x={W / 2} y={332} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">
        Zero init MLP 最后一层 → 初始 γ ≈ 1, β ≈ 0, α ≈ 0 → adaLN 退化成恒等映射
      </text>
      <text x={W / 2} y={350} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#10b981">
        训练从第 1 步就稳定 · 不需要 LR warmup · MLP 逐步学到合适调节量
      </text>
    </svg>
  );
}
