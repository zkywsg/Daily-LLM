import { STORAGE_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

export function StorageCompareChart() {
  const PAD_L = 170;
  const PAD_R = 80;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 60;

  const maxVal = 12;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="9 个 GLUE 任务存储对比">
      <text x={W / 2} y={18} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        9 个 GLUE 任务总存储:Full Fine-Tuning vs Adapter Tuning
      </text>

      {STORAGE_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const barW = Math.max(wOf(row.totalGB), 6);
        return (
          <g key={row.method}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={11} fontWeight={700} fill={row.color}>
              {row.method}
            </text>
            <rect x={PAD_L} y={y} width={barW} height={26} fill={row.bg} stroke={row.color} strokeWidth={1.6} rx={4} />
            <text x={PAD_L + barW + 8} y={y + 18} fontSize={12} fontWeight={700} fill={row.color}>
              {row.totalGB}GB
            </text>
            <text x={PAD_L} y={y + 42} fontSize={9} fill="var(--ink-muted)">{row.detail}</text>
          </g>
        );
      })}

      <text x={PAD_L} y={H - 6} fontSize={9} fill="var(--ink-muted)" fontStyle="italic">
        ↑ Adapter 只多存 8MB/任务的小模块,base 340M 权重跨任务共享 — 存储压缩约 9×
      </text>
    </svg>
  );
}
