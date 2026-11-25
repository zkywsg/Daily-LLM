import { ginPreMlp } from "../lib/data";

interface Props {
  epsilon: number;
}

const SELF_FEATURE = 1.0;
const NEIGHBOR_SUM = 2.4;

const W = 500;
const H = 260;

export function SelfVsNeighborBarWidget({ epsilon }: Props) {
  const selfContribution = (1 + epsilon) * SELF_FEATURE;
  const total = ginPreMlp(SELF_FEATURE, NEIGHBOR_SUM, epsilon);

  const PAD = { left: 60, right: 20, top: 40, bottom: 40 };
  const maxH = H - PAD.top - PAD.bottom;
  const maxVal = 6;
  const barW = 70;

  const bar = (x: number, val: number, label: string, color: string) => {
    const h = Math.min((val / maxVal) * maxH, maxH);
    return (
      <g key={label}>
        <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} fill={color} rx={3} />
        <text x={x + barW / 2} y={H - PAD.bottom - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
          {val.toFixed(2)}
        </text>
        <text x={x + barW / 2} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
          {label}
        </text>
      </g>
    );
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`ε=${epsilon} 时自身与邻居贡献对比`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        (1+ε)·h_self + Σneighbors,ε = {epsilon.toFixed(2)}
      </text>
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
      {bar(PAD.left + 20, selfContribution, "自身贡献", "#ec4899")}
      {bar(PAD.left + 20 + barW + 30, NEIGHBOR_SUM, "邻居贡献(固定)", "#9ca3af")}
      {bar(PAD.left + 20 + 2 * (barW + 30), total, "MLP 前总和", "#3b82f6")}
    </svg>
  );
}
