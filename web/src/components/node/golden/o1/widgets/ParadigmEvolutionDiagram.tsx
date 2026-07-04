import { PARADIGM_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  highlightIdx: number;
}

export function ParadigmEvolutionDiagram({ highlightIdx }: Props) {
  const colW = 210;
  const gap = 20;
  const startX = 30;
  const maxLogTokens = Math.log10(10000);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CoT / Self-Consistency / o1 三代范式对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        三代 reasoning 范式(GSM8K 演示)— "想"从 prompt 技巧变成模型权重
      </text>

      {PARADIGM_COMPARE.map((row, i) => {
        const x = startX + i * (colW + gap);
        const isFocus = i === highlightIdx || highlightIdx === -1;
        const barH = (Math.log10(row.thinkTokens) / maxLogTokens) * 120;
        const color = i === 0 ? "#f59e0b" : i === 1 ? "#3b82f6" : "#10b981";
        const bg = i === 0 ? "#fef3c7" : i === 1 ? "#dbeafe" : "#ecfdf5";
        return (
          <g key={row.stage} opacity={isFocus ? 1 : 0.3}>
            <text x={x + colW / 2} y={44} textAnchor="middle" fontSize={12} fontWeight={700} fill={color}>{row.stage}</text>

            <rect x={x + colW / 2 - 22} y={190 - barH} width={44} height={barH} fill={bg} stroke={color} strokeWidth={1.6} rx={3} />
            <text x={x + colW / 2} y={190 - barH - 8} textAnchor="middle" fontSize={10} fontWeight={700} fill={color}>
              {row.thinkTokens.toLocaleString()} tok
            </text>

            <text x={x + colW / 2} y={210} textAnchor="middle" fontSize={16} fontWeight={700} fill={color}>{row.accuracy}%</text>
            <text x={x + colW / 2} y={224} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">准确率</text>

            <foreignObject x={x} y={234} width={colW} height={30}>
              <div style={{ fontSize: 9, color: "var(--ink-muted)", textAlign: "center", lineHeight: 1.3 }}>{row.desc}</div>
            </foreignObject>
          </g>
        );
      })}
    </svg>
  );
}
