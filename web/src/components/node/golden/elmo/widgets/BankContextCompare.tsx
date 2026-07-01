import { CONTEXT_EXAMPLES, SIM_STATIC, SIM_ELMO } from "../lib/data";

const W = 700;
const H = 380;

interface Props {
  mode: "static" | "elmo";
}

// 左半:句子 1 (river bank) 走过 encoder,右半:句子 2 (money bank)
// mode = static: 两个 "bank" 向量完全相同(用同一颜色 dot)
// mode = elmo: 两个 "bank" 走不同 LSTM 路径 → 向量不同

export function BankContextCompare({ mode }: Props) {
  const TILE_W = 62;
  const TILE_H = 26;
  const GAP = 4;
  const yTop = 60;
  const yBot = 180;
  const startX = 30;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Static vs contextualized bank embedding">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "static"
          ? "静态词向量:'bank' 在两个句子里完全一样"
          : "ELMo:'bank' 走过不同 LSTM 路径 → 向量不同"}
      </text>

      {CONTEXT_EXAMPLES.map((ex, exIdx) => {
        const y = exIdx === 0 ? yTop : yBot;
        return (
          <g key={exIdx}>
            {/* label */}
            <text x={startX - 4} y={y - 6} fontSize={10} fontWeight={700} fill={ex.color}>
              句子 {exIdx + 1}: "{ex.meaning === "river" ? "河岸" : "银行"}"
            </text>

            {/* tokens */}
            {ex.sentence.map((tok, i) => {
              const x = startX + i * (TILE_W + GAP);
              const isBank = i === ex.bankIdx;
              return (
                <g key={i}>
                  <rect x={x} y={y} width={TILE_W} height={TILE_H}
                        rx={3}
                        fill={isBank ? "#fce7f3" : "#f3f4f6"}
                        stroke={isBank ? "#ec4899" : "#d1d5db"} strokeWidth={isBank ? 2 : 1} />
                  <text x={x + TILE_W / 2} y={y + TILE_H / 2 + 4}
                        textAnchor="middle" fontSize={10}
                        fontWeight={isBank ? 700 : 500} fill="#1f2937">
                    {tok}
                  </text>
                </g>
              );
            })}

            {/* encoder box */}
            {mode === "elmo" && (
              <g>
                <line x1={startX + ex.bankIdx * (TILE_W + GAP) + TILE_W / 2}
                      y1={y + TILE_H}
                      x2={startX + ex.bankIdx * (TILE_W + GAP) + TILE_W / 2}
                      y2={y + TILE_H + 15}
                      stroke={ex.color} strokeWidth={1.5} />
                <rect x={startX + ex.bankIdx * (TILE_W + GAP) + TILE_W / 2 - 45}
                      y={y + TILE_H + 15}
                      width={90} height={22}
                      rx={3} fill={ex.color} fillOpacity={0.15} stroke={ex.color} strokeWidth={1} />
                <text x={startX + ex.bankIdx * (TILE_W + GAP) + TILE_W / 2}
                      y={y + TILE_H + 30}
                      textAnchor="middle" fontSize={9} fontWeight={700} fill={ex.color}>
                  biLSTM 路径
                </text>
              </g>
            )}

            {/* 向量表示(dot cluster) */}
            <g transform={`translate(${startX + ex.sentence.length * (TILE_W + GAP) + 30}, ${y + TILE_H / 2})`}>
              <text x={0} y={-14} fontSize={9} fill="#6b7280">bank 向量</text>
              {mode === "static" ? (
                // 静态:两个 dot 完全一样,位置也相同(用同一色)
                [0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                  <circle key={i} cx={i * 8} cy={4} r={3} fill="#ec4899" />
                ))
              ) : (
                // ELMo:根据 meaning 不同用不同 pattern
                [0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                  <circle key={i} cx={i * 8}
                          cy={ex.meaning === "river" ? 4 - (i % 3) * 3 : 4 + (i % 2) * 4}
                          r={3} fill={ex.color} />
                ))
              )}
            </g>
          </g>
        );
      })}

      {/* similarity 展示 */}
      <rect x={80} y={290} width={W - 160} height={60} rx={6}
            fill={mode === "static" ? "#fce7f3" : "#ecfdf5"}
            stroke={mode === "static" ? "#ec4899" : "#10b981"} strokeWidth={1.5} />

      <text x={W / 2} y={314} textAnchor="middle" fontSize={11} fontWeight={700} fill="#374151">
        cosine_similarity(bank_1, bank_2)
      </text>
      <text x={W / 2} y={340} textAnchor="middle" fontSize={22} fontWeight={700}
            fill={mode === "static" ? "#831843" : "#065f46"}>
        {mode === "static" ? SIM_STATIC.toFixed(2) : SIM_ELMO.toFixed(2)}
      </text>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {mode === "static"
          ? "静态词向量 = 查表 · 无法区分 'river bank' 和 'money bank'"
          : "ELMo = 从上下文 LSTM 计算 · 两个 'bank' 向量清晰区分"}
      </text>
    </svg>
  );
}
