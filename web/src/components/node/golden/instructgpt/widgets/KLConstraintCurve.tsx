import { KL_CURVE } from "../lib/data";

interface Props {
  betaIdx: number;
}

const W = 700;
const H = 300;
const PAD = { left: 60, right: 80, top: 36, bottom: 50 };

// KL 系数 β 对 reward vs drift 的权衡。
// 太小 → reward hacking(模型为了高分输出胡言乱语)
// 太大 → policy 跟 SFT 一模一样,没收益

export function KLConstraintCurve({ betaIdx }: Props) {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const logMin = Math.log10(0.001);
  const logMax = Math.log10(1);
  const xScale = (b: number) => PAD.left + ((Math.log10(b) - logMin) / (logMax - logMin)) * innerW;

  const maxReward = 10;
  const yScale = (r: number) => PAD.top + (1 - r / maxReward) * innerH;

  const rewardPts = KL_CURVE.map((p) => `${xScale(p.beta)},${yScale(p.reward)}`).join(" ");
  const driftPts = KL_CURVE.map((p) => `${xScale(p.beta)},${yScale(p.drift * 10)}`).join(" ");

  const current = KL_CURVE[betaIdx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="KL constraint trade-off">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        KL 系数 β 的权衡 — reward vs drift
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* x 刻度 */}
      {KL_CURVE.map((p) => (
        <g key={p.beta}>
          <line x1={xScale(p.beta)} y1={H - PAD.bottom} x2={xScale(p.beta)} y2={H - PAD.bottom + 4} stroke="var(--ink-muted)" />
          <text x={xScale(p.beta)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {p.beta}
          </text>
        </g>
      ))}
      <text x={W / 2 - 20} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        β (KL 系数,log) →
      </text>

      {/* y 刻度 */}
      {[0, 5, 10].map((r) => (
        <g key={r}>
          <line x1={PAD.left - 4} x2={W - PAD.right} y1={yScale(r)} y2={yScale(r)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 8} y={yScale(r) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">{r}</text>
        </g>
      ))}

      {/* 危险区(reward hacking) */}
      <rect x={xScale(0.001)} y={PAD.top} width={xScale(0.005) - xScale(0.001)} height={innerH} fill="#fef2f2" opacity={0.7} />
      <text x={xScale(0.001) + 8} y={PAD.top + 14} fontSize={9} fontStyle="italic" fill="#7f1d1d">
        reward hacking
      </text>

      {/* reward 曲线 */}
      <polyline fill="none" stroke="#10b981" strokeWidth={2.4} points={rewardPts} />
      {KL_CURVE.map((p) => (
        <circle key={`r-${p.beta}`} cx={xScale(p.beta)} cy={yScale(p.reward)} r={4} fill={p.hacked ? "#dc2626" : "#10b981"} />
      ))}

      {/* drift 曲线(×10 缩放到同尺度) */}
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.4} strokeDasharray="4 3" points={driftPts} />
      {KL_CURVE.map((p) => (
        <circle key={`d-${p.beta}`} cx={xScale(p.beta)} cy={yScale(p.drift * 10)} r={4} fill="#ec4899" />
      ))}

      {/* 当前 β 高亮 */}
      <line x1={xScale(current.beta)} y1={PAD.top} x2={xScale(current.beta)} y2={H - PAD.bottom} stroke="#3b82f6" strokeWidth={1.6} strokeDasharray="3 3" />
      <text x={xScale(current.beta)} y={PAD.top - 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="#3b82f6">
        β = {current.beta}
      </text>

      {/* 图例 */}
      <g transform={`translate(${W - PAD.right + 4}, ${PAD.top + 12})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#10b981" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fill="#10b981">reward</text>
        </g>
        <g transform="translate(0, 16)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={2.4} strokeDasharray="4 3" />
          <text x={18} y={4} fontSize={10} fill="#ec4899">drift ×10</text>
        </g>
      </g>
    </svg>
  );
}
