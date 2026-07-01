import { BENCHMARK_RESULTS } from "../lib/data";

const W = 700;
const H = 380;

const CATEGORY_COLOR: Record<string, string> = {
  nli: "#ec4899",
  reading: "#10b981",
  similarity: "#f59e0b",
  sentiment: "#3b82f6",
  grammar: "#a855f7",
};

export function BenchmarkGainBars() {
  const PAD_L = 90;
  const PAD_R = 60;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 24;
  const gap = 4;

  const maxGain = 12;
  const minGain = -8;
  const zeroX = PAD_L + (0 - minGain) / (maxGain - minGain) * plotW;
  const xOf = (v: number) => PAD_L + (v - minGain) / (maxGain - minGain) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-1 12 benchmark gains">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        12 个 Benchmark 相对前作 SOTA 的提升(9/12 SOTA)
      </text>

      <line x1={zeroX} y1={PAD_T - 6} x2={zeroX} y2={PAD_T + BENCHMARK_RESULTS.length * (rowH + gap)} stroke="#9ca3af" strokeWidth={1.2} />

      {BENCHMARK_RESULTS.map((r, i) => {
        const y = PAD_T + i * (rowH + gap);
        const color = CATEGORY_COLOR[r.category];
        const isNeg = r.gain < 0;
        const barX = isNeg ? xOf(r.gain) : zeroX;
        const barW = Math.abs(xOf(r.gain) - zeroX);
        return (
          <g key={r.task}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={10} fontWeight={600} fill="#374151">{r.task}</text>
            <rect x={barX} y={y} width={Math.max(barW, 1)} height={rowH - 4}
                  fill={color} fillOpacity={isNeg ? 0.2 : 0.25} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={isNeg ? barX - 6 : barX + barW + 6} y={y + rowH / 2 + 4}
                  textAnchor={isNeg ? "end" : "start"}
                  fontSize={10} fontWeight={700} fill={color}>
              {r.gain >= 0 ? "+" : ""}{r.gain.toFixed(1)}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        长上下文推理类(QNLI/StoryCloze/RACE)提升最大 — 印证 BookCorpus 长程依赖训练的价值
      </text>
    </svg>
  );
}
