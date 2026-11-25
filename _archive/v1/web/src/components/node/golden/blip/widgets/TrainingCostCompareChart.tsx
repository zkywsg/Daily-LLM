import { TRAINING_COST_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

export function TrainingCostCompareChart() {
  const PAD_L = 140;
  const PAD_T = 40;
  const plotW = 420;
  const rowH = 60;
  const maxLog = Math.log10(20000);
  const wOf = (v: number) => (Math.log10(v) / maxLog) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Flamingo vs BLIP-2 训练成本对比(log 尺度)">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        训练算力(GPU 天,log 尺度)— 降低约 100×
      </text>

      {TRAINING_COST_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isBlip = row.model === "BLIP-2";
        const color = isBlip ? "#10b981" : "#9ca3af";
        const bg = isBlip ? "#ecfdf5" : "#f3f4f6";
        return (
          <g key={row.model}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={11} fontWeight={700} fill={color}>{row.model}</text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.gpuDays), 4)} height={22} fill={bg} stroke={color} strokeWidth={1.6} rx={4} />
            <text x={PAD_L + Math.max(wOf(row.gpuDays), 4) + 8} y={y + 17} fontSize={11} fontWeight={700} fill={color}>
              {row.gpuDays.toLocaleString()} GPU 天 · ${row.costUSD.toLocaleString()}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
