import { TASK_GAINS } from "../lib/data";

const W = 700;
const H = 340;

export function TaskGainBars() {
  const PAD_L = 90;
  const PAD_R = 40;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;

  const maxVal = 100;
  const wOf = (v: number) => (v / maxVal) * plotW;

  const rowH = 20;
  const groupGap = 8;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ELMo 6 task gains">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        6 个 NLP 任务:baseline / prev SOTA / + ELMo(全部 SOTA)
      </text>

      {/* legend */}
      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">baseline</text>
        <rect x={100} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={118} y={10} fontSize={10} fill="#374151">prev SOTA</text>
        <rect x={220} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={238} y={10} fontSize={10} fill="#374151">+ ELMo</text>
      </g>

      {TASK_GAINS.map((row, i) => {
        const y = PAD_T + i * (rowH * 3 + groupGap);
        return (
          <g key={row.task}>
            <text x={PAD_L - 8} y={y + rowH + rowH / 2 + 4} textAnchor="end"
                  fontSize={11} fontWeight={700} fill="#374151">
              {row.task}
              <tspan fontSize={9} fill="#9ca3af"> ({row.metric})</tspan>
            </text>

            {/* baseline */}
            <rect x={PAD_L} y={y} width={wOf(row.baseline)} height={rowH - 2}
                  fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} rx={2} />
            <text x={PAD_L + wOf(row.baseline) + 4} y={y + rowH / 2 + 4}
                  fontSize={9} fill="#9ca3af">{row.baseline.toFixed(1)}</text>

            {/* prev sota */}
            <rect x={PAD_L} y={y + rowH} width={wOf(row.prevSota)} height={rowH - 2}
                  fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} rx={2} />
            <text x={PAD_L + wOf(row.prevSota) + 4} y={y + rowH + rowH / 2 + 4}
                  fontSize={9} fill="#3b82f6">{row.prevSota.toFixed(1)}</text>

            {/* + ELMo */}
            <rect x={PAD_L} y={y + rowH * 2} width={wOf(row.withElmo)} height={rowH - 2}
                  fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.withElmo) + 4} y={y + rowH * 2 + rowH / 2 + 4}
                  fontSize={9} fontWeight={700} fill="#ec4899">{row.withElmo.toFixed(1)}</text>

            {/* gain */}
            <text x={W - 8} y={y + rowH + rowH / 2 + 4} textAnchor="end"
                  fontSize={10} fontWeight={700} fill="#065f46">
              +{(row.withElmo - row.baseline).toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
