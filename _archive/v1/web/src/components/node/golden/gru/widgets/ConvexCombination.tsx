const W = 700;
const H = 300;

interface Props {
  z: number; // 0..1
}

// 展示凸组合 h_t = (1-z)*h_prev + z*h_cand,用两个水平条(旧状态/新候选)按比例混合
export function ConvexCombination({ z }: Props) {
  const barW = 500;
  const barX = (W - barW) / 2;
  const oldW = (1 - z) * barW;
  const newW = z * barW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GRU update gate convex combination">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        更新门凸组合 — h_t = (1−z)⊙h_{"{t-1}"} + z⊙h̃_t
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        z = {z.toFixed(2)} · (1−z) + z = 1 自动满足,不需要独立学习两个门
      </text>

      {/* 条形混合 */}
      <rect x={barX} y={80} width={oldW} height={50} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} />
      <rect x={barX + oldW} y={80} width={newW} height={50} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />

      <text x={barX + oldW / 2} y={110} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">
        {oldW > 60 ? `旧状态 (1-z)=${(1 - z).toFixed(2)}` : ""}
      </text>
      <text x={barX + oldW + newW / 2} y={110} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">
        {newW > 60 ? `新候选 z=${z.toFixed(2)}` : ""}
      </text>

      {/* 对比 LSTM 的独立门 */}
      <text x={W / 2} y={165} textAnchor="middle" fontSize={11} fontWeight={700} fill="#374151">
        对比 LSTM(独立 forget f + input i,无凸约束)
      </text>
      <rect x={barX} y={180} width={barW * 0.6} height={30} fill="#dbeafe" fillOpacity={0.5} stroke="#3b82f6" strokeDasharray="3 3" />
      <rect x={barX} y={220} width={barW * 0.5} height={30} fill="#fce7f3" fillOpacity={0.5} stroke="#ec4899" strokeDasharray="3 3" />
      <text x={barX + barW * 0.6 + 8} y={200} fontSize={10} fill="#1e40af">f (可独立设为任意值)</text>
      <text x={barX + barW * 0.5 + 8} y={240} fontSize={10} fill="#ec4899">i (可独立设为任意值)</text>
      <text x={barX} y={268} fontSize={10} fontStyle="italic" fill="#9ca3af">
        LSTM 可以同时 f=0,i=0(清空状态)— GRU 凸组合做不到这一点
      </text>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        z≈0 完全保留旧状态 · z≈1 完全采用新候选 · 中间是加权平均
      </text>
    </svg>
  );
}
