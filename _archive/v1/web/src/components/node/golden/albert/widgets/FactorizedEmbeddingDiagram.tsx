import { rawEmbeddingParams, factorizedEmbeddingParams } from "../lib/data";

interface Props {
  embedSize: number; // E,可调
}

const VOCAB = 30000;
const HIDDEN = 1024; // BERT-large hidden size,作为演示基准

export function FactorizedEmbeddingDiagram({ embedSize }: Props) {
  const W = 700;
  const H = 300;

  const rawParams = rawEmbeddingParams(VOCAB, HIDDEN); // V × H
  const factParams = factorizedEmbeddingParams(VOCAB, embedSize, HIDDEN); // V×E + E×H

  const maxParams = rawParams;
  const PAD_L = 160;
  const plotW = W - PAD_L - 60;
  const wOf = (v: number) => (v / maxParams) * plotW;

  const rows = [
    { label: `原版 V × H (H=${HIDDEN})`, value: rawParams, color: "#3b82f6", bg: "#dbeafe" },
    { label: `因式分解 V×E + E×H (E=${embedSize})`, value: factParams, color: "#10b981", bg: "#ecfdf5" },
  ];

  const reduction = (1 - factParams / rawParams) * 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Embedding 因式分解参数量对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Token Embedding 参数量:V × H vs V × E + E × H
      </text>

      {rows.map((row, i) => {
        const y = 70 + i * 90;
        return (
          <g key={row.label}>
            <text x={PAD_L - 10} y={y + 20} textAnchor="end" fontSize={11} fontWeight={700} fill={row.color}>
              {row.label}
            </text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.value), 4)} height={28} fill={row.bg} stroke={row.color} strokeWidth={1.6} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.value), 4) + 8} y={y + 19} fontSize={11} fontWeight={700} fill={row.color}>
              {(row.value / 1e6).toFixed(1)}M
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={270} textAnchor="middle" fontSize={12} fontWeight={700} fill="#10b981">
        E = {embedSize} 时:参数减少 {reduction.toFixed(0)}%({(rawParams / 1e6).toFixed(1)}M → {(factParams / 1e6).toFixed(1)}M)
      </text>
      <text x={W / 2} y={288} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        vocab_size = 30K,H = 1024(BERT-large 基准)— E 越小,压缩越狠,但 E 太小会损失 token 表达力
      </text>
    </svg>
  );
}
