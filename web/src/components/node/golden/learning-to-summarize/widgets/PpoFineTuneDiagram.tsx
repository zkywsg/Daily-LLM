import { KL_TRADEOFF } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  betaIdx: number;
}

export function PpoFineTuneDiagram({ betaIdx }: Props) {
  const row = KL_TRADEOFF[betaIdx];

  const plotX = 90;
  const plotW = 500;
  const plotBottom = 300;
  const plotTop = 170;
  const maxVal = 10;
  const xOf = (i: number) => plotX + (i / (KL_TRADEOFF.length - 1)) * plotW;
  const yOf = (v: number) => plotBottom - (v / maxVal) * (plotBottom - plotTop);

  const rewardPath = KL_TRADEOFF.map((r, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(r.rewardScore)}`).join(" ");
  const klPath = KL_TRADEOFF.map((r, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(r.klDrift)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PPO + KL 惩罚微调示意图">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        policy 生成 → RM 打分 → PPO 更新,KL 惩罚拉回参考模型
      </text>

      {/* pipeline */}
      <g fontSize={9}>
        <rect x={30} y={40} width={110} height={44} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} />
        <text x={85} y={58} textAnchor="middle" fontWeight={700} fill="#1e3a8a">π_θ(policy)</text>
        <text x={85} y={72} textAnchor="middle" fill="#1e3a8a">生成摘要 y</text>

        <line x1={140} y1={62} x2={168} y2={62} stroke="var(--border)" strokeWidth={1.5} markerEnd="url(#arrow-ppo)" />
        <defs>
          <marker id="arrow-ppo" markerWidth={8} markerHeight={8} refX={4} refY={4} orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--border)" />
          </marker>
        </defs>

        <rect x={168} y={40} width={110} height={44} rx={6} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.2} />
        <text x={223} y={58} textAnchor="middle" fontWeight={700} fill="#9d174d">Reward Model</text>
        <text x={223} y={72} textAnchor="middle" fill="#9d174d">r(x, y) 打分</text>

        <line x1={278} y1={62} x2={306} y2={62} stroke="var(--border)" strokeWidth={1.5} markerEnd="url(#arrow-ppo)" />

        <rect x={306} y={40} width={130} height={44} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} />
        <text x={371} y={58} textAnchor="middle" fontWeight={700} fill="#92400e">PPO 更新 π_θ</text>
        <text x={371} y={72} textAnchor="middle" fill="#92400e">max r − β·KL</text>

        {/* KL branch to reference model */}
        <rect x={306} y={100} width={130} height={40} rx={6} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} strokeDasharray="4,3" />
        <text x={371} y={116} textAnchor="middle" fontWeight={700} fill="#374151">π_SFT(参考,冻结)</text>
        <text x={371} y={130} textAnchor="middle" fill="#374151">KL(π_θ ‖ π_SFT)</text>
        <line x1={371} y1={100} x2={371} y2={84} stroke="#9ca3af" strokeWidth={1.4} strokeDasharray="4,3" markerEnd="url(#arrow-ppo)" />
      </g>

      {/* KL tradeoff curve */}
      <line x1={plotX} y1={plotBottom} x2={plotX + plotW} y2={plotBottom} stroke="var(--border)" strokeWidth={1} />
      <line x1={plotX} y1={plotTop} x2={plotX} y2={plotBottom} stroke="var(--border)" strokeWidth={1} />
      <text x={plotX - 10} y={plotTop + 6} textAnchor="end" fontSize={8} fill="var(--ink-muted)">高</text>
      <text x={plotX - 10} y={plotBottom} textAnchor="end" fontSize={8} fill="var(--ink-muted)">低</text>

      <path d={rewardPath} fill="none" stroke="#10b981" strokeWidth={2} />
      <path d={klPath} fill="none" stroke="#ef4444" strokeWidth={2} strokeDasharray="4,3" />

      {KL_TRADEOFF.map((r, i) => (
        <g key={i}>
          <circle cx={xOf(i)} cy={yOf(r.rewardScore)} r={i === betaIdx ? 5 : 3} fill={i === betaIdx ? "#059669" : "#10b981"} />
          <circle cx={xOf(i)} cy={yOf(r.klDrift)} r={i === betaIdx ? 5 : 3} fill={i === betaIdx ? "#b91c1c" : "#ef4444"} />
          <text x={xOf(i)} y={plotBottom + 16} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">β={r.beta}</text>
        </g>
      ))}

      <g transform={`translate(${plotX + plotW - 120}, ${plotTop - 10})`} fontSize={9}>
        <line x1={0} y1={0} x2={16} y2={0} stroke="#10b981" strokeWidth={2} />
        <text x={20} y={3} fill="#374151">reward(RM 打分)</text>
        <line x1={0} y1={14} x2={16} y2={14} stroke="#ef4444" strokeWidth={2} strokeDasharray="4,3" />
        <text x={20} y={17} fill="#374151">KL(π_θ‖π_SFT)</text>
      </g>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontWeight={700} fill={row.hacked ? "#b91c1c" : "#065f46"}>
        当前 β = {row.beta}:reward = {row.rewardScore},KL = {row.klDrift} — {row.hacked ? "⚠ β 太小,reward hacking 风险高" : "✓ 稳定,policy 未偏离太远"}
      </text>
    </svg>
  );
}
