import { ZERO_SHOT_RESULTS } from "../lib/data";

const W = 700;
const H = 320;

// 3 列条形:prev zero-shot · GPT-2 XL zero-shot · supervised SOTA
export function ZeroShotVsSftBars() {
  const PAD_L = 70;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 70;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const n = ZERO_SHOT_RESULTS.length;
  const groupW = plotW / n;
  const barW = 20;
  const gap = 2;

  const yOf = (acc: number) => PAD_T + ((100 - acc) / 100) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-2 zero-shot vs supervised SOTA">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Zero-shot 性能对比 — 前作 / GPT-2 XL / 监督 SOTA(论文 Table 3)
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 25, 50, 75, 100].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}%</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {ZERO_SHOT_RESULTS.map((r, i) => {
        const groupX = PAD_L + i * groupW + groupW / 2 - (3 * barW + 2 * gap) / 2;
        const vals = [
          { label: "prev", v: r.prev_zs,  color: "#9ca3af", bg: "#f3f4f6" },
          { label: "GPT-2", v: r.gpt2_xl, color: "#ec4899", bg: "#fce7f3" },
          { label: "SFT",  v: r.sft_sota, color: "#3b82f6", bg: "#dbeafe" },
        ];
        return (
          <g key={i}>
            {vals.map((b, k) => {
              if (b.v === null) return null;
              const x = groupX + k * (barW + gap);
              const yy = yOf(b.v);
              return (
                <g key={k}>
                  <rect x={x} y={yy} width={barW} height={(PAD_T + plotH) - yy} fill={b.bg} stroke={b.color} strokeWidth={1.2} rx={2} />
                  <text x={x + barW / 2} y={yy - 4} textAnchor="middle" fontSize={9} fontWeight={600} fill={b.color}>{b.v.toFixed(1)}</text>
                </g>
              );
            })}
            <text x={groupX + (3 * barW + 2 * gap) / 2} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">{r.task}</text>
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${PAD_L + 20}, ${H - 30})`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">前作 zero-shot</text>
        <rect x={120} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={138} y={10} fontSize={10} fill="#374151">GPT-2 XL zero-shot</text>
        <rect x={280} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={298} y={10} fontSize={10} fill="#374151">监督学习 SOTA</text>
      </g>
    </svg>
  );
}
