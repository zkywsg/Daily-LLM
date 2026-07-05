import { REPRO_RATE } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  withBn: boolean;
}

// 生成一条模拟的训练损失曲线(有 BN 时平滑收敛,没有 BN 时剧烈震荡/崩溃)
function buildLossPath(withBn: boolean, plotW: number, plotH: number): string {
  const points: [number, number][] = [];
  const n = 40;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    let y: number;
    if (withBn) {
      // 平滑指数衰减 + 小噪声
      y = 0.9 * Math.exp(-3 * t) + 0.08 + 0.015 * Math.sin(i * 1.7);
    } else {
      // 前段震荡剧烈,后段可能崩到 0 或炸掉(mode collapse / D 完胜)
      const noise = Math.sin(i * 0.9) * 0.35 * Math.exp(-t * 0.5);
      const spike = i > 26 ? (i % 4 === 0 ? 0.5 : -0.15) : 0;
      y = Math.max(0.02, 0.55 + noise + spike * (1 - t));
    }
    const x = t * plotW;
    const yy = plotH - Math.min(1, Math.max(0, y)) * plotH;
    points.push([x, yy]);
  }
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
}

export function BatchNormEffectDiagram({ withBn }: Props) {
  const PAD_L = 50;
  const PAD_T = 50;
  const plotW = 300;
  const plotH = 180;

  const path = buildLossPath(withBn, plotW, plotH);
  const color = withBn ? "#10b981" : "#ec4899";
  const bg = withBn ? "#ecfdf5" : "#fce7f3";

  const barMaxW = 260;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="BatchNorm 对 GAN 训练稳定性的影响"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {withBn ? "有 BatchNorm — 训练损失平滑收敛" : "没有 BatchNorm — 训练损失剧烈震荡 / 崩溃"}
      </text>

      <g transform={`translate(${PAD_L}, ${PAD_T})`}>
        <rect x={-10} y={-10} width={plotW + 20} height={plotH + 30} fill={bg} rx={6} opacity={0.5} />
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="var(--border)" strokeWidth={1} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke="var(--border)" strokeWidth={1} />
        <path d={path} fill="none" stroke={color} strokeWidth={2} />
        <text x={plotW / 2} y={plotH + 20} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
          训练 iteration →
        </text>
        <text x={-8} y={-4} fontSize={9} fill="var(--ink-muted)" textAnchor="end">
          D/G loss
        </text>
      </g>

      <g transform={`translate(${PAD_L + plotW + 60}, ${PAD_T})`}>
        <text x={0} y={-16} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
          GAN 复现成功率
        </text>
        {REPRO_RATE.map((row, i) => {
          const y = i * 44;
          const w = (row.successRate / 100) * barMaxW;
          const fill = row.label.includes("之后") ? "#10b981" : "#9ca3af";
          return (
            <g key={row.label} transform={`translate(0, ${y})`}>
              <text x={0} y={0} fontSize={9} fill="var(--ink-secondary)">{row.label}</text>
              <rect x={0} y={6} width={barMaxW} height={16} fill="#f3f4f6" stroke="var(--border)" rx={3} />
              <rect x={0} y={6} width={w} height={16} fill={fill} rx={3} />
              <text x={w + 6} y={18} fontSize={10} fontWeight={700} fill="var(--ink-primary)">
                {row.successRate}%
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
