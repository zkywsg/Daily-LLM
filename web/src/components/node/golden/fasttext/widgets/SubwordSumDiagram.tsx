import { getSubwords } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  word: string;
  showOov?: boolean;
}

export function SubwordSumDiagram({ word, showOov = false }: Props) {
  const subwords = getSubwords(word).slice(0, 9); // 演示用截断
  const boxW = 62;
  const gap = 6;
  const totalW = subwords.length * (boxW + gap) - gap;
  const startX = (W - totalW) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`${word} 的词向量 = subword 向量之和`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {showOov ? `"${word}"(未见过整词)仍可从 subword 组合出向量` : `v("${word}") = Σ z_subword`}
      </text>

      {subwords.map((g, i) => {
        const x = startX + i * (boxW + gap);
        const known = !showOov || i % 4 !== 3; // 演示:偶尔标一个 subword 也是罕见但仍参与训练
        return (
          <g key={i}>
            <rect x={x} y={50} width={boxW} height={34} fill={known ? "#fef3c7" : "#fce7f3"} stroke={known ? "#f59e0b" : "#ec4899"} strokeWidth={1.4} rx={4} />
            <text x={x + boxW / 2} y={71} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#374151">{g}</text>
            <text x={x + boxW / 2} y={100} textAnchor="middle" fontSize={9} fill="#9ca3af">z_{i + 1}</text>
            {i < subwords.length - 1 && (
              <text x={x + boxW + gap / 2} y={71} textAnchor="middle" fontSize={14} fill="#9ca3af">+</text>
            )}
          </g>
        );
      })}

      <line x1={W / 2 - 90} y1={140} x2={W / 2 + 90} y2={140} stroke="#374151" strokeWidth={1.5} />
      <path d={`M ${W / 2 + 84} 136 L ${W / 2 + 90} 140 L ${W / 2 + 84} 144`} fill="none" stroke="#374151" strokeWidth={1.5} />

      <rect x={W / 2 - 60} y={155} width={120} height={40} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.6} rx={6} />
      <text x={W / 2} y={172} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">v("{word}")</text>
      <text x={W / 2} y={187} textAnchor="middle" fontSize={9} fill="#065f46">{subwords.length} 个向量求和</text>

      <text x={W / 2} y={225} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {showOov
          ? "即使整词从未在训练语料出现,只要它的 subword 出现过,依然能合成合理向量"
          : "每个 subword 有自己的向量,词向量是所有 subword 向量的和(可选加整词向量)"}
      </text>
    </svg>
  );
}
