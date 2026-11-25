import { emergentDiscreteAcc, emergentContinuousScore } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  metric: "discrete" | "continuous" | "both";
}

// 同一任务在 0/1 acc vs token-level acc 两个指标下的曲线
export function EmergenceDebate({ metric }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xMin = 22, xMax = 25;
  const xOf = (lf: number) => PAD_L + ((lf - xMin) / (xMax - xMin)) * plotW;
  const yOf = (p: number) => PAD_T + (1 - p / 100) * plotH;

  const discretePts: string[] = [];
  const continuousPts: string[] = [];
  for (let i = 0; i <= 200; i++) {
    const lf = xMin + (i / 200) * (xMax - xMin);
    discretePts.push(`${xOf(lf)},${yOf(emergentDiscreteAcc(lf))}`);
    continuousPts.push(`${xOf(lf)},${yOf(emergentContinuousScore(lf))}`);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Emergence discrete vs continuous metric">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        涌现是真的还是评测伪影? — 同任务两种指标对比
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 25, 50, 75, 100].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}%</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}
      {[22, 23, 24, 25].map((lf) => (
        <g key={lf}>
          <line x1={xOf(lf)} y1={PAD_T + plotH} x2={xOf(lf)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(lf)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">10^{lf}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">training FLOPs (log)</text>

      {/* 阈值线 */}
      <line x1={xOf(24)} y1={PAD_T} x2={xOf(24)} y2={PAD_T + plotH} stroke="#ec4899" strokeDasharray="3 3" />
      <text x={xOf(24) + 6} y={PAD_T + 14} fontSize={10} fontWeight={700} fill="#ec4899">"涌现阈值"</text>

      {(metric === "discrete" || metric === "both") && (
        <polyline points={discretePts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.5} />
      )}
      {(metric === "continuous" || metric === "both") && (
        <polyline points={continuousPts.join(" ")} fill="none" stroke="#10b981" strokeWidth={2.5} />
      )}

      {/* legend */}
      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 6})`}>
        <g opacity={metric === "continuous" ? 0.4 : 1}>
          <line x1={0} y1={6} x2={20} y2={6} stroke="#ec4899" strokeWidth={2.5} />
          <text x={26} y={10} fontSize={10} fontWeight={600} fill="#ec4899">0/1 acc(离散) — 跳变</text>
        </g>
        <g transform="translate(220, 0)" opacity={metric === "discrete" ? 0.4 : 1}>
          <line x1={0} y1={6} x2={20} y2={6} stroke="#10b981" strokeWidth={2.5} />
          <text x={26} y={10} fontSize={10} fontWeight={600} fill="#10b981">token-level(连续) — 平滑</text>
        </g>
      </g>

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        Schaeffer 2023:涌现"突然性"被评测离散性夸大 · 连续指标看到的是平滑增长
      </text>
    </svg>
  );
}
