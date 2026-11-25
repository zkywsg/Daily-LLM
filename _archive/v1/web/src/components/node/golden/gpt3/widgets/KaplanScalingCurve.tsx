import { MODEL_LINEUP, scalingLoss, fmtParams } from "../lib/scaling";

interface Props {
  currentParams: number;
}

const W = 700;
const H = 320;
const PAD = { left: 64, right: 24, top: 36, bottom: 50 };

// log-log loss vs params 曲线,标 GPT 系列点位 + 当前 slider 位置。
// 让 viewer 看到 Kaplan power-law 是怎么外推到 175B 的。
export function KaplanScalingCurve({ currentParams }: Props) {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const minLogN = 7; // 10M
  const maxLogN = 13; // 10T
  const xScale = (logN: number) => PAD.left + ((logN - minLogN) / (maxLogN - minLogN)) * innerW;

  // loss 区间在 ~1.5 → ~3.5
  const minLoss = 1.5;
  const maxLoss = 3.5;
  const yScale = (loss: number) => PAD.top + ((loss - minLoss) / (maxLoss - minLoss)) * innerH;

  // 拟合曲线点(连续)
  const curvePts: string[] = [];
  for (let logN = minLogN; logN <= maxLogN; logN += 0.1) {
    const N = Math.pow(10, logN);
    const loss = scalingLoss(N);
    curvePts.push(`${xScale(logN)},${yScale(loss)}`);
  }

  const currentLogN = Math.log10(currentParams);
  const currentLoss = scalingLoss(currentParams);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Kaplan scaling law: loss vs params">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Kaplan Scaling Law:loss ≈ (N_c / N)^α_N
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* x 轴标签 (log scale) */}
      {[8, 9, 10, 11, 12, 13].map((logN) => (
        <g key={`x-${logN}`}>
          <line x1={xScale(logN)} y1={H - PAD.bottom} x2={xScale(logN)} y2={H - PAD.bottom + 4} stroke="var(--ink-muted)" />
          <text x={xScale(logN)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
            10^{logN}
          </text>
        </g>
      ))}
      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        参数量 N (log scale) →
      </text>

      {/* y 轴标签 */}
      {[1.5, 2.0, 2.5, 3.0, 3.5].map((l) => (
        <g key={`y-${l}`}>
          <line x1={PAD.left - 4} x2={W - PAD.right} y1={yScale(l)} y2={yScale(l)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 8} y={yScale(l) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {l.toFixed(1)}
          </text>
        </g>
      ))}
      <text x={20} y={PAD.top + innerH / 2} textAnchor="middle" fontSize={11} fill="var(--ink-muted)" transform={`rotate(-90 20 ${PAD.top + innerH / 2})`}>
        loss
      </text>

      {/* 拟合曲线 */}
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.5} points={curvePts.join(" ")} />

      {/* 模型族系点 */}
      {MODEL_LINEUP.map((m) => {
        const logN = Math.log10(m.params);
        const loss = scalingLoss(m.params);
        return (
          <g key={m.name}>
            <circle cx={xScale(logN)} cy={yScale(loss)} r={6} fill={m.color} stroke="var(--bg-canvas)" strokeWidth={2} />
            <text x={xScale(logN)} y={yScale(loss) - 12} textAnchor="middle" fontSize={10} fontWeight={600} fill={m.color}>
              {m.name}
            </text>
          </g>
        );
      })}

      {/* 当前 slider 位置 */}
      <line
        x1={xScale(currentLogN)}
        y1={PAD.top}
        x2={xScale(currentLogN)}
        y2={H - PAD.bottom}
        stroke="#10b981"
        strokeWidth={1.5}
        strokeDasharray="4 3"
      />
      <circle cx={xScale(currentLogN)} cy={yScale(currentLoss)} r={8} fill="#10b981" stroke="var(--bg-canvas)" strokeWidth={2} />
      <text x={xScale(currentLogN) + 12} y={yScale(currentLoss) + 4} fontSize={11} fontWeight={700} fill="#10b981">
        N = {fmtParams(currentParams)} · loss ≈ {currentLoss.toFixed(2)}
      </text>
    </svg>
  );
}
