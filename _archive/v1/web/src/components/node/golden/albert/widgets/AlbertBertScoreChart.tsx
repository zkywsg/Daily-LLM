import { ALBERT_BERT_TABLE } from "../lib/data";

const W = 700;
const H = 420;

export function AlbertBertScoreChart() {
  const PAD_L = 150;
  const PAD_R = 60;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 56;

  const maxParams = 340;
  const wOf = (v: number) => (v / maxParams) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ALBERT vs BERT 参数量与性能对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        参数量 vs 性能:ALBERT-xxlarge(235M)反超 BERT-large(334M)
      </text>

      {ALBERT_BERT_TABLE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isAlbert = row.model.startsWith("ALBERT");
        const isXXLarge = row.model === "ALBERT-xxlarge";
        const color = isXXLarge ? "#10b981" : isAlbert ? "#f59e0b" : "#3b82f6";
        const bg = isXXLarge ? "#ecfdf5" : isAlbert ? "#fef3c7" : "#dbeafe";
        return (
          <g key={row.model}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>
              {row.model}
            </text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.paramsM), 4)} height={20} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.paramsM), 4) + 6} y={y + 15} fontSize={10} fontWeight={700} fill={color}>
              {row.paramsM}M
            </text>
            <text x={PAD_L} y={y + 34} fontSize={9} fill="var(--ink-muted)">
              SQuAD F1 {row.squadF1} · MNLI {row.mnli} · RACE {row.race} · 训练 {row.trainHours}h
            </text>
          </g>
        );
      })}
    </svg>
  );
}
