import { MORPHOLOGY_COMPARE } from "../lib/data";

const W = 700;
const H = 300;

export function MorphologyCompareBars() {
  const PAD_L = 90;
  const PAD_R = 40;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 42;

  const maxVal = 55;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="FastText vs Word2Vec 形态丰富语言对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        形态越丰富的语言,FastText 优势越大
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">Word2Vec</text>
        <rect x={100} y={0} width={12} height={12} fill="#fef3c7" stroke="#f59e0b" />
        <text x={118} y={10} fontSize={10} fill="#374151">FastText</text>
      </g>

      {MORPHOLOGY_COMPARE.map((row, i) => {
        const y = PAD_T + i * (rowH + 4);
        return (
          <g key={row.lang}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.lang}</text>

            <rect x={PAD_L} y={y} width={wOf(row.word2vec)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.word2vec) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.word2vec.toFixed(1)}</text>

            <rect x={PAD_L} y={y + 18} width={wOf(row.fasttext)} height={16} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.fasttext) + 6} y={y + 31} fontSize={9} fontWeight={700} fill="#b45309">
              {row.fasttext.toFixed(1)}(+{row.delta.toFixed(1)})
            </text>
          </g>
        );
      })}
    </svg>
  );
}
