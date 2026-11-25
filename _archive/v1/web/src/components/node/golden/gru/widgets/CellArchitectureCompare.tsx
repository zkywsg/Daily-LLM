const W = 700;
const H = 360;

interface Props {
  side: "lstm" | "gru" | "both";
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

export function CellArchitectureCompare({ side }: Props) {
  const dimL = side === "gru" ? 0.35 : 1;
  const dimG = side === "lstm" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LSTM vs GRU cell architecture">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LSTM(4 门 + 双状态)vs GRU(2 门 + 单一状态)
      </text>

      {/* === LSTM 左 === */}
      <g opacity={dimL}>
        <text x={175} y={46} textAnchor="middle" fontSize={12} fontWeight={700} fill="#831843">LSTM</text>

        <Box x={30}  y={60} w={70} h={30} fill="#fef3c7" stroke="#f59e0b" label="forget f" />
        <Box x={110} y={60} w={70} h={30} fill="#fce7f3" stroke="#ec4899" label="input i" />
        <Box x={190} y={60} w={70} h={30} fill="#dbeafe" stroke="#3b82f6" label="output o" />
        <Box x={110} y={100} w={70} h={30} fill="#ecfdf5" stroke="#10b981" label="C̃ candidate" />

        <Box x={30} y={160} w={130} h={36} fill="#fce7f3" stroke="#ec4899" label="C_t (cell state)" sub="长期 highway" />
        <Box x={30} y={210} w={130} h={36} fill="#dbeafe" stroke="#3b82f6" label="h_t (hidden)" sub="短期接口" />

        <text x={175} y={270} textAnchor="middle" fontSize={10} fill="#374151">4 组矩阵 · 双状态</text>
        <text x={175} y={288} textAnchor="middle" fontSize={10} fontWeight={700} fill="#831843">参数 4d(d+x)</text>
        <text x={175} y={306} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#9ca3af">f,i 独立学习</text>
      </g>

      {/* 分隔线 */}
      <line x1={350} y1={40} x2={350} y2={320} stroke="#e5e7eb" strokeDasharray="3 3" />

      {/* === GRU 右 === */}
      <g opacity={dimG} transform="translate(370, 0)">
        <text x={175} y={46} textAnchor="middle" fontSize={12} fontWeight={700} fill="#065f46">GRU</text>

        <Box x={60}  y={60} w={90} h={30} fill="#fef3c7" stroke="#f59e0b" label="reset r" />
        <Box x={170} y={60} w={90} h={30} fill="#fce7f3" stroke="#ec4899" label="update z" />
        <Box x={110} y={100} w={100} h={30} fill="#ecfdf5" stroke="#10b981" label="h̃ candidate" />

        <Box x={60} y={160} w={200} h={36} fill="#dbeafe" stroke="#3b82f6" label="h_t (单一状态)" sub="合并长短期" />

        <text x={175} y={220} textAnchor="middle" fontSize={10} fill="#374151">3 组矩阵 · 单一状态</text>
        <text x={175} y={238} textAnchor="middle" fontSize={10} fontWeight={700} fill="#065f46">参数 3d(d+x) · 省 25%</text>
        <text x={175} y={256} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#9ca3af">(1-z)+z=1 凸组合</text>
      </g>

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        GRU 把 forget+input 绑成凸组合 · 把 C 和 h 合并 · 训练快 15-20%(非 25%,因候选要等 r 算完才能算)
      </text>
    </svg>
  );
}
