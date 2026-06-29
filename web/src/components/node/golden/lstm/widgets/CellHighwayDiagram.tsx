const W = 700;
const H = 260;

// 横向高速路 + 时间步:cell state C_{t-1} → ⊙f → +i⊙g → C_t (一直贯穿)
// 强调 highway 上只有 element-wise multiply 和 add,梯度反向传播时不被 W 反复乘。

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
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

export function CellHighwayDiagram() {
  const hwY = 90;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Cell state highway across timesteps">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Cell State 高速路:C_{"ₜ₋₁"} ─⊙f─ ─+i⊙g─ C_t (一条直线贯穿)
      </text>

      {/* 高速路本体 */}
      <line x1={20} y1={hwY} x2={W - 20} y2={hwY} stroke="#ec4899" strokeWidth={6} opacity={0.85} />

      {/* 入口 C_{t-1} */}
      <Box x={20} y={hwY - 22} w={70} h={44} fill="#fce7f3" stroke="#ec4899" label="C_{ₜ₋₁}" sub="上一步 cell" />

      {/* ⊙ f_t */}
      <circle cx={170} cy={hwY} r={18} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={170} y={hwY + 4} textAnchor="middle" fontSize={14} fontWeight={700} fill="#92400e">⊙</text>
      <text x={170} y={hwY + 40} textAnchor="middle" fontSize={11} fontWeight={600} fill="#92400e">f_t</text>
      <text x={170} y={hwY + 54} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#6b7280">forget gate</text>

      {/* + i⊙g (来自下方) */}
      <circle cx={320} cy={hwY} r={18} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={320} y={hwY + 4} textAnchor="middle" fontSize={14} fontWeight={700} fill="#1e3a8a">+</text>
      <line x1={320} y1={hwY + 18} x2={320} y2={hwY + 50} stroke="#3b82f6" strokeWidth={1.5} markerEnd="url(#up-arr)" />
      <defs>
        <marker id="up-arr" viewBox="0 0 10 10" refX="5" refY="0" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 10 L 5 0 L 10 10" fill="none" stroke="#3b82f6" strokeWidth={1} />
        </marker>
      </defs>
      <Box x={285} y={hwY + 50} w={70} h={34} fill="#dbeafe" stroke="#3b82f6" label="i_t ⊙ g_t" sub="新写入" />

      {/* 出口 C_t */}
      <Box x={W - 90} y={hwY - 22} w={70} h={44} fill="#fce7f3" stroke="#ec4899" label="C_t" sub="当前 cell" />

      {/* tanh + ⊙ o → h_t (从 highway 分叉出去) */}
      <line x1={450} y1={hwY} x2={450} y2={170} stroke="#ec4899" strokeWidth={1.5} strokeDasharray="3 3" />
      <Box x={470} y={150} w={70} h={34} fill="#ecfdf5" stroke="#10b981" label="tanh" />
      <line x1={540} y1={167} x2={570} y2={167} stroke="#10b981" strokeWidth={1.5} markerEnd="url(#right-arr)" />
      <defs>
        <marker id="right-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
        </marker>
      </defs>
      <Box x={570} y={150} w={70} h={34} fill="#ecfdf5" stroke="#10b981" label="⊙ o_t" sub="→ h_t" />

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        反向传播沿粉色 highway 走只乘 f_t(≈0.95)和加常数,不被矩阵 W 反复挤压
      </text>
    </svg>
  );
}
