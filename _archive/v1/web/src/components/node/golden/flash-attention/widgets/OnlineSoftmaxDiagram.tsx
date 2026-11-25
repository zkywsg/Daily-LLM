import { simulateOnlineSoftmax } from "../lib/data";

const W = 700;
const H = 300;

const DEMO_BLOCKS: number[][] = [
  [1.2, 0.4, -0.8],
  [2.5, 1.9, 0.3],
  [-0.5, 0.1, 1.0],
];

interface Props {
  visibleBlocks: number;
}

export function OnlineSoftmaxDiagram({ visibleBlocks }: Props) {
  const steps = simulateOnlineSoftmax(DEMO_BLOCKS.slice(0, visibleBlocks));
  const maxM = 3;
  const chartW = 460;
  const chartH = 120;
  const PAD_L = 160;
  const PAD_T = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="online softmax 增量更新演示">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Online Softmax — 每来一个新 block 就 rescale 一次
      </text>

      {DEMO_BLOCKS.map((scores, bi) => {
        const seen = bi < visibleBlocks;
        const y = 40;
        const x = 40 + bi * 150;
        return (
          <g key={bi} opacity={seen ? 1 : 0.3}>
            <text x={x + 60} y={y - 6} textAnchor="middle" fontSize={10} fontWeight={700} fill="#6b7280">block {bi}</text>
            {scores.map((s, si) => (
              <g key={si}>
                <rect x={x + si * 42} y={y} width={36} height={24} fill={seen ? "#dbeafe" : "var(--bg-surface)"} stroke={seen ? "#3b82f6" : "var(--border)"} rx={3} />
                <text x={x + si * 42 + 18} y={y + 16} textAnchor="middle" fontSize={10} fill="#1e40af">{s.toFixed(1)}</text>
              </g>
            ))}
          </g>
        );
      })}

      <g transform={`translate(${PAD_L}, ${PAD_T + 70})`}>
        <line x1={0} y1={0} x2={chartW} y2={0} stroke="var(--border)" strokeWidth={1} />
        <text x={-12} y={4} textAnchor="end" fontSize={9} fill="var(--ink-muted)">m/l</text>

        {steps.map((step, i) => {
          const x = (i / Math.max(DEMO_BLOCKS.length - 1, 1)) * chartW;
          const yM = -Math.min(step.mNew, maxM) * (chartH / maxM);
          const yL = -Math.min(Math.log(step.lNew + 1) * 20, chartH);
          return (
            <g key={i}>
              <circle cx={x} cy={yM} r={5} fill="#f59e0b" />
              <text x={x} y={yM - 10} textAnchor="middle" fontSize={9} fontWeight={700} fill="#b45309">m={step.mNew.toFixed(2)}</text>
              <circle cx={x} cy={yL} r={5} fill="#ec4899" />
              <text x={x} y={yL + 18} textAnchor="middle" fontSize={9} fontWeight={700} fill="#be185d">ℓ={step.lNew.toFixed(2)}</text>
              {i > 0 && (
                <>
                  <line
                    x1={(x - chartW / Math.max(DEMO_BLOCKS.length - 1, 1))}
                    y1={-Math.min(steps[i - 1].mNew, maxM) * (chartH / maxM)}
                    x2={x}
                    y2={yM}
                    stroke="#f59e0b"
                    strokeWidth={1.4}
                    strokeDasharray="3 2"
                  />
                </>
              )}
            </g>
          );
        })}
      </g>

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        running max m_i(橙)只增不减,running sum ℓ_i(粉,log 尺度显示)每步用 rescale 与新 max 对齐 — 全程不需要看完整行
      </text>
    </svg>
  );
}
