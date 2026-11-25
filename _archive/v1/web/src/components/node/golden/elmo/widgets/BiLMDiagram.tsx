const W = 700;
const H = 380;

interface Props {
  highlight: "char" | "fwd" | "bwd" | "concat" | null;
}

// biLM 结构:char-CNN → 2 层 biLSTM (fwd + bwd 独立)

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

const TOKENS = ["I", "walked", "along", "the", "river", "bank"];

export function BiLMDiagram({ highlight }: Props) {
  const dimChar = highlight && highlight !== "char" ? 0.35 : 1;
  const dimFwd = highlight && highlight !== "fwd" ? 0.35 : 1;
  const dimBwd = highlight && highlight !== "bwd" ? 0.35 : 1;
  const dimC = highlight && highlight !== "concat" ? 0.35 : 1;

  const cellW = 78;
  const gap = 6;
  const startX = 40;
  const yToken = 340;
  const yChar = 290;
  const yL1F = 230;
  const yL1B = 200;
  const yL2F = 140;
  const yL2B = 110;
  const yOut = 50;

  const xOf = (i: number) => startX + i * (cellW + gap) + cellW / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ELMo biLM architecture">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Deep biLM — char-CNN + 2 层双向 LSTM
      </text>

      {/* Tokens */}
      {TOKENS.map((t, i) => (
        <g key={i}>
          <rect x={startX + i * (cellW + gap)} y={yToken} width={cellW} height={24}
                fill="#f3f4f6" stroke="#d1d5db" strokeWidth={1} rx={3} />
          <text x={xOf(i)} y={yToken + 16} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{t}</text>
        </g>
      ))}

      {/* char-CNN */}
      <g opacity={dimChar}>
        {TOKENS.map((_, i) => (
          <g key={i}>
            <line x1={xOf(i)} y1={yToken} x2={xOf(i)} y2={yChar + 24} stroke="#f59e0b" strokeWidth={1.2} />
            <Box x={startX + i * (cellW + gap)} y={yChar} w={cellW} h={24}
                 fill="#fef3c7" stroke="#f59e0b" label={`char-CNN`} />
          </g>
        ))}
        <text x={W - 20} y={yChar + 16} fontSize={10} fontWeight={700} fill="#92400e">Layer 0</text>
      </g>

      {/* LSTM L1 forward + backward */}
      <g opacity={dimFwd}>
        {TOKENS.map((_, i) => (
          <g key={i}>
            <line x1={xOf(i)} y1={yChar} x2={xOf(i)} y2={yL1F + 24} stroke="#ec4899" strokeWidth={1.2} />
            <Box x={startX + i * (cellW + gap)} y={yL1F} w={cellW} h={22}
                 fill="#fce7f3" stroke="#ec4899" label={`L1 →`} />
            {/* horizontal fwd arrow */}
            {i < TOKENS.length - 1 && (
              <line x1={startX + i * (cellW + gap) + cellW} y1={yL1F + 11}
                    x2={startX + (i + 1) * (cellW + gap)} y2={yL1F + 11}
                    stroke="#ec4899" strokeWidth={1.2} />
            )}
          </g>
        ))}
      </g>

      <g opacity={dimBwd}>
        {TOKENS.map((_, i) => (
          <g key={i}>
            <Box x={startX + i * (cellW + gap)} y={yL1B} w={cellW} h={22}
                 fill="#dbeafe" stroke="#3b82f6" label={`L1 ←`} />
            {i < TOKENS.length - 1 && (
              <line x1={startX + i * (cellW + gap) + cellW} y1={yL1B + 11}
                    x2={startX + (i + 1) * (cellW + gap)} y2={yL1B + 11}
                    stroke="#3b82f6" strokeWidth={1.2} strokeDasharray="3 2" />
            )}
          </g>
        ))}
        <text x={W - 20} y={yL1F + 22} fontSize={10} fontWeight={700} fill="#831843">Layer 1</text>
      </g>

      {/* LSTM L2 forward + backward */}
      <g opacity={dimFwd}>
        {TOKENS.map((_, i) => (
          <g key={i}>
            <line x1={xOf(i)} y1={yL1B} x2={xOf(i)} y2={yL2F + 24} stroke="#ec4899" strokeWidth={1.2} />
            <Box x={startX + i * (cellW + gap)} y={yL2F} w={cellW} h={22}
                 fill="#fce7f3" stroke="#ec4899" label={`L2 →`} />
            {i < TOKENS.length - 1 && (
              <line x1={startX + i * (cellW + gap) + cellW} y1={yL2F + 11}
                    x2={startX + (i + 1) * (cellW + gap)} y2={yL2F + 11}
                    stroke="#ec4899" strokeWidth={1.2} />
            )}
          </g>
        ))}
      </g>

      <g opacity={dimBwd}>
        {TOKENS.map((_, i) => (
          <g key={i}>
            <Box x={startX + i * (cellW + gap)} y={yL2B} w={cellW} h={22}
                 fill="#dbeafe" stroke="#3b82f6" label={`L2 ←`} />
            {i < TOKENS.length - 1 && (
              <line x1={startX + i * (cellW + gap) + cellW} y1={yL2B + 11}
                    x2={startX + (i + 1) * (cellW + gap)} y2={yL2B + 11}
                    stroke="#3b82f6" strokeWidth={1.2} strokeDasharray="3 2" />
            )}
          </g>
        ))}
        <text x={W - 20} y={yL2F + 22} fontSize={10} fontWeight={700} fill="#831843">Layer 2</text>
      </g>

      {/* Concat 输出 */}
      <g opacity={dimC}>
        {TOKENS.map((_, i) => (
          <g key={i}>
            <line x1={xOf(i)} y1={yL2B} x2={xOf(i)} y2={yOut + 24} stroke="#10b981" strokeWidth={1.2} />
            <Box x={startX + i * (cellW + gap)} y={yOut} w={cellW} h={24}
                 fill="#ecfdf5" stroke="#10b981" label={`h_${i} (2L+1)`} />
          </g>
        ))}
        <text x={W - 20} y={yOut + 16} fontSize={10} fontWeight={700} fill="#065f46">输出</text>
      </g>
    </svg>
  );
}
