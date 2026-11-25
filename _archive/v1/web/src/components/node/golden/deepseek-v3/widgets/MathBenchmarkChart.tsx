import { BENCHMARK_TABLE } from "../lib/data";

const W = 700;
const H = 340;

// MATH 基准分数对比 —— V3 最亮眼的一条:90.2 反超所有闭源旗舰。
export function MathBenchmarkChart() {
  const PAD = { left: 170, right: 60, top: 50, bottom: 30 };
  const innerW = W - PAD.left - PAD.right;
  const rowH = 48;
  const max = 100;

  const xScale = (v: number) => (v / max) * innerW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="MATH 基准分数对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        MATH 基准分数(技术报告 Table 4)
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        DeepSeek-V3 90.2 —— 开源第一次在数学能力上反超闭源旗舰
      </text>

      {BENCHMARK_TABLE.map((row, i) => {
        const y = PAD.top + i * rowH;
        const w = xScale(row.math);
        const isV3 = !!row.highlight;
        return (
          <g key={row.model}>
            <text x={PAD.left - 10} y={y + 20} textAnchor="end" fontSize={11} fontWeight={isV3 ? 700 : 500} fill={isV3 ? "#6366f1" : "var(--ink-primary)"}>
              {row.model}
            </text>
            <rect x={PAD.left} y={y + 4} width={w} height={24} rx={4} fill={isV3 ? "#6366f1" : "#c7d2fe"} opacity={isV3 ? 0.9 : 0.6} />
            <text x={PAD.left + w + 8} y={y + 21} fontSize={11} fontWeight={700} fill={isV3 ? "#4338ca" : "var(--ink-secondary)"}>
              {row.math.toFixed(1)}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        对照 Claude-3.5-Sonnet 78.3 · GPT-4o 76.6 · LLaMA-3.1-405B 73.8
      </text>
    </svg>
  );
}
