import { getSubwords } from "../lib/data";

const W = 700;

interface Props {
  word: string;
}

const N_LEVELS = [3, 4, 5, 6];

export function NgramDecomposeDiagram({ word }: Props) {
  const bounded = `<${word}>`;
  const rowH = 30;
  const H = 60 + N_LEVELS.length * rowH + 50;
  const charW = Math.min(28, (W - 80) / bounded.length);
  const PAD_L = 40;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`${word} 的 subword n-gram 分解`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        "{word}" → n-gram 分解(n=3..6)+ 整词 token
      </text>

      <g transform={`translate(${PAD_L}, 40)`}>
        {bounded.split("").map((ch, i) => (
          <g key={i}>
            <rect x={i * charW} y={0} width={charW - 2} height={22} fill="#dbeafe" stroke="#3b82f6" rx={3} />
            <text x={i * charW + charW / 2 - 1} y={16} textAnchor="middle" fontSize={12} fontWeight={700} fill="#1e40af">{ch}</text>
          </g>
        ))}
      </g>

      {N_LEVELS.map((n, rowIdx) => {
        const y = 40 + 32 + rowIdx * rowH;
        if (n > bounded.length) return null;
        const starts: number[] = [];
        for (let i = 0; i <= bounded.length - n; i++) starts.push(i);
        return (
          <g key={n} transform={`translate(${PAD_L}, ${y})`}>
            <text x={-14} y={16} textAnchor="end" fontSize={10} fontWeight={700} fill="#6b7280">n={n}</text>
            {starts.map((s) => (
              <rect
                key={s}
                x={s * charW}
                y={2}
                width={n * charW - 4}
                height={18}
                fill="#fce7f3"
                stroke="#ec4899"
                strokeWidth={1}
                rx={3}
                opacity={0.85}
              />
            ))}
          </g>
        );
      })}

      <g transform={`translate(${PAD_L}, ${40 + 32 + N_LEVELS.length * rowH + 10})`}>
        <rect x={0} y={0} width={bounded.length * charW - 2} height={20} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={3} />
        <text x={(bounded.length * charW - 2) / 2} y={14} textAnchor="middle" fontSize={10} fontWeight={700} fill="#065f46">
          整词 token {bounded}
        </text>
      </g>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        共 {getSubwords(word).length} 个 subword(含整词 token)
      </text>
    </svg>
  );
}
