const W = 700;
const H = 360;

// 完整 LSTM cell 详细图:输入 [h_{t-1}, x_t] → 4 个线性变换 → forget/input/output gate + candidate
// 然后 C 高速路 + h 输出。
// 用三种 gate 配色:forget 黄 / input 粉 / output 绿 / cell highway 粉

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color }: { x1: number; y1: number; x2: number; y2: number; color?: string }) {
  const c = color ?? "#9ca3af";
  const id = `gate-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={c} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function ThreeGatesDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LSTM three gates detailed diagram">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LSTM Cell 完整结构:forget / input / output 三门 + candidate
      </text>

      {/* 输入 [h_{t-1}, x_t] */}
      <Box x={20} y={170} w={110} h={40} fill="#fef3c7" stroke="#f59e0b" label="[h_{ₜ₋₁}; x_t]" sub="concat" />

      {/* 4 个线性变换分别走 4 个门 */}
      {/* forget */}
      <Arrow x1={130} y1={185} x2={200} y2={70} color="#f59e0b" />
      <Box x={200} y={50} w={130} h={36} fill="#fef3c7" stroke="#f59e0b" label="f_t = σ(W_f·[h,x])" />

      {/* input */}
      <Arrow x1={130} y1={185} x2={200} y2={130} color="#ec4899" />
      <Box x={200} y={110} w={130} h={36} fill="#fce7f3" stroke="#ec4899" label="i_t = σ(W_i·[h,x])" />

      {/* candidate */}
      <Arrow x1={130} y1={195} x2={200} y2={200} color="#3b82f6" />
      <Box x={200} y={180} w={130} h={36} fill="#dbeafe" stroke="#3b82f6" label="g_t = tanh(W_g·[h,x])" />

      {/* output */}
      <Arrow x1={130} y1={210} x2={200} y2={300} color="#10b981" />
      <Box x={200} y={280} w={130} h={36} fill="#ecfdf5" stroke="#10b981" label="o_t = σ(W_o·[h,x])" />

      {/* 上一步 C_{t-1} 进入 */}
      <Box x={20} y={70} w={90} h={36} fill="#fce7f3" stroke="#ec4899" label="C_{ₜ₋₁}" />
      <Arrow x1={110} y1={88} x2={360} y2={88} color="#ec4899" />

      {/* ⊙ f */}
      <circle cx={350} cy={88} r={14} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={350} y={92} textAnchor="middle" fontSize={12} fontWeight={700} fill="#92400e">⊙</text>
      <Arrow x1={335} y1={75} x2={335} y2={88} color="#f59e0b" />

      {/* + i⊙g */}
      <circle cx={420} cy={88} r={14} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={420} y={92} textAnchor="middle" fontSize={12} fontWeight={700} fill="#831843">+</text>
      <Arrow x1={335} y1={150} x2={400} y2={102} color="#ec4899" />
      <Arrow x1={335} y1={210} x2={400} y2={102} color="#3b82f6" />

      <Arrow x1={420} y1={88} x2={580} y2={88} color="#ec4899" />
      <Box x={580} y={70} w={90} h={36} fill="#fce7f3" stroke="#ec4899" label="C_t" />

      {/* h = o ⊙ tanh(C) */}
      <line x1={500} y1={88} x2={500} y2={250} stroke="#ec4899" strokeWidth={1.4} strokeDasharray="3 3" />
      <Box x={460} y={250} w={70} h={30} fill="#ecfdf5" stroke="#10b981" label="tanh" />
      <circle cx={550} cy={265} r={14} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
      <text x={550} y={269} textAnchor="middle" fontSize={12} fontWeight={700} fill="#065f46">⊙</text>
      <Arrow x1={335} y1={300} x2={538} y2={272} color="#10b981" />
      <Arrow x1={530} y1={265} x2={580} y2={265} color="#10b981" />
      <Box x={580} y={245} w={90} h={36} fill="#ecfdf5" stroke="#10b981" label="h_t" />

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        forget 控保留多少旧 C · input 控写入多少新 g · output 控暴露多少 cell 给 h
      </text>
    </svg>
  );
}
