import { PARAM_COMPARE, SPEED_COMPARE } from "../lib/data";

const W = 700;
const H = 300;

export function ParamSpeedBars() {
  const PAD_L = 110;
  const PAD_R = 60;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 30;

  const maxParam = Math.max(...PARAM_COMPARE.map((p) => p.params));
  const wOf = (v: number) => (v / maxParam) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Parameter and speed comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        参数量对比(d=x=256)与训练速度
      </text>

      {PARAM_COMPARE.map((row, i) => {
        const y = PAD_T + i * (rowH + 10);
        const color = row.unit === "GRU" ? "#10b981" : row.unit === "LSTM" ? "#ec4899" : "#9ca3af";
        const bg = row.unit === "GRU" ? "#ecfdf5" : row.unit === "LSTM" ? "#fce7f3" : "#f3f4f6";
        return (
          <g key={row.unit}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.unit}</text>
            <rect x={PAD_L} y={y} width={wOf(row.params)} height={rowH - 4} fill={bg} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.params) + 6} y={y + rowH / 2 + 4} fontSize={10} fontWeight={700} fill={color}>
              {row.params.toFixed(0)}K ({row.gates} 组矩阵)
            </text>
          </g>
        );
      })}

      {/* speed comparison bottom */}
      <text x={W / 2} y={PAD_T + 3 * (rowH + 10) + 24} textAnchor="middle" fontSize={11} fontWeight={700} fill="#374151">
        相对 LSTM(100%)的训练速度 与 参数量
      </text>

      <g transform={`translate(${PAD_L}, ${PAD_T + 3 * (rowH + 10) + 40})`}>
        <rect x={0} y={0} width={(SPEED_COMPARE.gru.params / 100) * 300} height={20}
              fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} />
        <text x={(SPEED_COMPARE.gru.params / 100) * 300 + 6} y={15} fontSize={10} fontWeight={700} fill="#10b981">
          GRU 参数 {SPEED_COMPARE.gru.params}%
        </text>
        <rect x={0} y={28} width={(SPEED_COMPARE.gru.trainSpeed / 100) * 300} height={20}
              fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} />
        <text x={(SPEED_COMPARE.gru.trainSpeed / 100) * 300 + 6} y={43} fontSize={10} fontWeight={700} fill="#3b82f6">
          GRU 训练速度 {SPEED_COMPARE.gru.trainSpeed}%(非 75% — 因需 2 次矩阵乘)
        </text>
      </g>
    </svg>
  );
}
