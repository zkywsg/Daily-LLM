import { PARADIGM_COMPARE } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightIdx: number;
}

export function ParadigmCompareChart({ highlightIdx }: Props) {
  const PAD_L = 170;
  const PAD_T = 50;
  const plotW = 380;
  const rowH = 50;
  const maxVal = 65;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Standard/CoT/Act-only/ReAct 范式对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        ReAct 全面胜过纯推理和纯行动 — 1+1&gt;2
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={16} y={9} fontSize={9} fill="#374151">HotpotQA EM</text>
        <rect x={110} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={126} y={9} fontSize={9} fill="#374151">Fever Acc</text>
      </g>

      {PARADIGM_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        const isReact = row.pattern === "ReAct";
        return (
          <g key={row.pattern} opacity={isFocus ? 1 : 0.3}>
            <text x={PAD_L - 10} y={y + 12} textAnchor="end" fontSize={10} fontWeight={700} fill={isReact ? "#065f46" : "#374151"}>{row.pattern}</text>
            <text x={PAD_L - 10} y={y + 24} textAnchor="end" fontSize={8} fill="var(--ink-muted)" fontFamily="monospace">{row.form}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.hotpotEM), 4)} height={14} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.hotpotEM), 4) + 6} y={y + 11} fontSize={9} fontWeight={700} fill="#1e40af">{row.hotpotEM}</text>

            <rect x={PAD_L} y={y + 16} width={Math.max(wOf(row.feverAcc), 4)} height={14} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.feverAcc), 4) + 6} y={y + 27} fontSize={9} fontWeight={700} fill="#065f46">{row.feverAcc}</text>
          </g>
        );
      })}
    </svg>
  );
}
