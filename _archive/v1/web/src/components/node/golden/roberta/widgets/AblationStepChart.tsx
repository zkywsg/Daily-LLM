import { ABLATION_STEPS } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  visibleSteps: number;
}

export function AblationStepChart({ visibleSteps }: Props) {
  const PAD_L = 40;
  const PAD_R = 30;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 46;

  const baseSquad = ABLATION_STEPS[0].squad;
  const baseMnli = ABLATION_STEPS[0].mnli;
  const maxDelta = 4; // 最大展示涨幅区间

  const wOf = (v: number, base: number) => (Math.max(v - base, 0) / maxDelta) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RoBERTa 五项改动累积消融">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        五项改动累积贡献 — "更多数据 + 更长训练"才是真正的决定因素
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={18} y={10} fontSize={9} fill="#374151">SQuAD F1(涨幅,相对 baseline)</text>
        <rect x={220} y={0} width={12} height={12} fill="#ecfdf5" stroke="#10b981" />
        <text x={238} y={10} fontSize={9} fill="#374151">MNLI(涨幅,相对 baseline)</text>
      </g>

      {ABLATION_STEPS.slice(0, visibleSteps).map((row, i) => {
        const y = PAD_T + i * (rowH + 4);
        const isLast = i === ABLATION_STEPS.length - 1;
        return (
          <g key={row.label}>
            <text x={0} y={y + rowH / 2 - 6} fontSize={10} fontWeight={700} fill={isLast ? "#065f46" : "#374151"}>{row.label}</text>

            <rect x={0} y={y} width={Math.max(wOf(row.squad, baseSquad), 2)} height={16} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={2} />
            <text x={Math.max(wOf(row.squad, baseSquad), 2) + 6} y={y + 13} fontSize={9} fontWeight={700} fill="#1e40af">
              {row.squad.toFixed(1)}(+{(row.squad - baseSquad).toFixed(1)})
            </text>

            <rect x={0} y={y + 18} width={Math.max(wOf(row.mnli, baseMnli), 2)} height={16} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={Math.max(wOf(row.mnli, baseMnli), 2) + 6} y={y + 31} fontSize={9} fontWeight={700} fill="#065f46">
              {row.mnli.toFixed(1)}(+{(row.mnli - baseMnli).toFixed(1)})
            </text>
          </g>
        );
      })}
    </svg>
  );
}
