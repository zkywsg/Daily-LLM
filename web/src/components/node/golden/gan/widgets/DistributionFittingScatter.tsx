import { useMemo } from "react";
import { sampleReal, sampleGen, REAL_MODES } from "../lib/dist";

interface Props {
  iter: number;
  totalIter: number;
  modeCollapse?: boolean;
}

const W = 700;
const H = 400;
const PAD = 36;

// 2D 散点图:粉色 = 真分布(4 模 GMM 固定),蓝色 = G 生成分布(随 iter 收敛)。
// 拖 iter slider:开始 G 还在乱采,后期 4 个真模周围聚出 4 个蓝团。
// modeCollapse=true 时 G 只学到第 1 个模 → viewer 一眼看到 mode collapse。

export function DistributionFittingScatter({ iter, totalIter, modeCollapse }: Props) {
  const realPts = useMemo(() => sampleReal(200, 42), []);
  const genPts = useMemo(() => sampleGen(200, iter, totalIter, 7, modeCollapse), [iter, totalIter, modeCollapse]);

  // 坐标映射:数据 x ∈ [-3, 3] → svg [PAD, W-PAD]
  const xScale = (x: number) => PAD + ((x + 3) / 6) * (W - 2 * PAD);
  const yScale = (y: number) => PAD + ((y + 3) / 6) * (H - 2 * PAD);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Distribution fitting, iter ${iter}/${totalIter}${modeCollapse ? ", mode collapse" : ""}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        2D 分布拟合 — iter {iter} / {totalIter}{modeCollapse ? " · mode collapse" : ""}
      </text>

      {/* 坐标轴 */}
      <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="var(--border)" strokeDasharray="2 3" />
      <line x1={W / 2} y1={PAD} x2={W / 2} y2={H - PAD} stroke="var(--border)" strokeDasharray="2 3" />

      {/* 真分布 4 模中心 */}
      {REAL_MODES.map((m, i) => (
        <circle
          key={`mode-${i}`}
          cx={xScale(m.cx)}
          cy={yScale(m.cy)}
          r={3 + m.std * 30}
          fill="none"
          stroke="#ec4899"
          strokeWidth={1}
          strokeDasharray="2 2"
          opacity={0.4}
        />
      ))}

      {/* 真分布点(粉色) */}
      {realPts.map((p, i) => (
        <circle
          key={`r-${i}`}
          cx={xScale(p.x)}
          cy={yScale(p.y)}
          r={2}
          fill="#ec4899"
          opacity={0.45}
        />
      ))}

      {/* G 生成点(蓝色) */}
      {genPts.map((p, i) => (
        <circle
          key={`g-${i}`}
          cx={xScale(p.x)}
          cy={yScale(p.y)}
          r={2}
          fill="#3b82f6"
          opacity={0.55}
        />
      ))}

      {/* 图例 */}
      <g transform={`translate(${PAD + 10}, ${H - PAD - 28})`}>
        <circle cx={6} cy={6} r={4} fill="#ec4899" opacity={0.5} />
        <text x={16} y={10} fontSize={11} fill="var(--ink-secondary)">p_data (真分布,4 模 GMM)</text>
        <g transform="translate(0, 14)">
          <circle cx={6} cy={6} r={4} fill="#3b82f6" opacity={0.6} />
          <text x={16} y={10} fontSize={11} fill="var(--ink-secondary)">p_g (G 生成分布)</text>
        </g>
      </g>
    </svg>
  );
}
