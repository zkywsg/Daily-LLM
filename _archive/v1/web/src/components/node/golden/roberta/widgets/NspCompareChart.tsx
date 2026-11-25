import { NSP_FORMAT_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  highlightIdx: number;
}

export function NspCompareChart({ highlightIdx }: Props) {
  const PAD_L = 40;
  const PAD_T = 50;
  const plotW = 500;
  const rowH = 60;
  const mnliMin = 86, mnliMax = 89;
  const squadMin = 90, squadMax = 93;
  const wOfMnli = (v: number) => ((v - mnliMin) / (mnliMax - mnliMin)) * plotW;
  const wOfSquad = (v: number) => ((v - squadMin) / (squadMax - squadMin)) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="NSP 输入格式对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        去 NSP + 单序列输入 — 反而更好
      </text>

      {NSP_FORMAT_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        const color = i === 2 ? "#10b981" : i === 1 ? "#3b82f6" : "#9ca3af";
        const bg = i === 2 ? "#ecfdf5" : i === 1 ? "#dbeafe" : "#f3f4f6";
        return (
          <g key={row.format} opacity={isFocus ? 1 : 0.3}>
            <text x={0} y={y - 6} fontSize={10} fontWeight={700} fill={color}>{row.format}</text>

            <rect x={0} y={y} width={Math.max(wOfMnli(row.mnli), 4)} height={16} fill={bg} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={Math.max(wOfMnli(row.mnli), 4) + 6} y={y + 13} fontSize={9} fontWeight={700} fill={color}>MNLI {row.mnli}</text>

            <rect x={0} y={y + 20} width={Math.max(wOfSquad(row.squad), 4)} height={16} fill={bg} stroke={color} strokeWidth={1.4} rx={2} opacity={0.7} />
            <text x={Math.max(wOfSquad(row.squad), 4) + 6} y={y + 33} fontSize={9} fontWeight={700} fill={color}>SQuAD {row.squad}</text>
          </g>
        );
      })}
    </svg>
  );
}
