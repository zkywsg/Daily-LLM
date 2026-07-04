import { SIMILARITY_COMPARE } from "../lib/data";

const W = 700;
const H = 300;

export function SimilarityCompareBars() {
  const PAD_L = 90;
  const PAD_R = 40;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 40;

  const maxVal = 90;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GloVe vs Word2Vec similarity benchmarks">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Word Similarity — GloVe 全面胜过 Word2Vec 5-10 点
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">Word2Vec SG</text>
        <rect x={110} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={128} y={10} fontSize={10} fill="#374151">GloVe</text>
      </g>

      {SIMILARITY_COMPARE.map((row, i) => {
        const y = PAD_T + i * (rowH + 4);
        return (
          <g key={row.task}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.task}</text>

            <rect x={PAD_L} y={y} width={wOf(row.word2vec)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.word2vec) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.word2vec.toFixed(1)}</text>

            <rect x={PAD_L} y={y + 18} width={wOf(row.glove)} height={16} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.glove) + 6} y={y + 31} fontSize={9} fontWeight={700} fill="#ec4899">{row.glove.toFixed(1)}</text>
          </g>
        );
      })}
    </svg>
  );
}
