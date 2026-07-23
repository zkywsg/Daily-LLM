import { rawNeighbors, attentionLogit, softmax, HEAD_SEEDS } from "../lib/data";

interface Props {
  center: number;
  mode: "concat" | "average";
}

const W = 700;
const H = 380;

// 4 个头并排展示各自的注意力权重分布(小型 bar group),
// 下方再画一条"组合后输出"的条形图:concat 模式下把 4 个头的
// 权重依次排开(输出维度变宽),average 模式下逐元素平均(维度不变)。

export function MultiHeadWidget({ center, mode }: Props) {
  const neighbors = rawNeighbors(center);
  const headWeights = HEAD_SEEDS.map((seed) => {
    const logits = neighbors.map((n) => attentionLogit(center, n, 1, seed));
    return softmax(logits);
  });

  const combined =
    mode === "average"
      ? neighbors.map((_, idx) => headWeights.reduce((s, hw) => s + hw[idx], 0) / headWeights.length)
      : headWeights.flat();

  const headColors = ["#ec4899", "#3b82f6", "#10b981", "#f59e0b"];
  const PAD = { left: 50, right: 20, top: 30, bottom: 30 };
  const rowH = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的多头注意力`}>
      {headWeights.map((weights, headIdx) => {
        const y = PAD.top + headIdx * rowH;
        const cellW = (W - PAD.left - PAD.right) / neighbors.length;
        return (
          <g key={headIdx}>
            <text x={PAD.left - 8} y={y + rowH / 2} textAnchor="end" fontSize={11} fontWeight={600} fill={headColors[headIdx]}>
              Head {headIdx}
            </text>
            {weights.map((w, idx) => {
              const h = Math.min(w * (rowH - 20) * 3, rowH - 20);
              const x = PAD.left + idx * cellW;
              return (
                <g key={idx}>
                  <rect x={x + 4} y={y + rowH - 10 - h} width={cellW - 8} height={h} fill={headColors[headIdx]} opacity={0.8} />
                  <text x={x + cellW / 2} y={y + rowH - 12 - h} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
                    {w.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}

      <text x={W / 2} y={PAD.top + 4 * rowH + 20} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        组合输出({mode === "concat" ? "concat,维度 = 4 × 邻居数" : "average,维度 = 邻居数"})
      </text>
      {combined.map((v, idx) => {
        const cellW = (W - PAD.left - PAD.right) / combined.length;
        const x = PAD.left + idx * cellW;
        const y0 = PAD.top + 4 * rowH + 30;
        const h = Math.min(v * 60 * 3, 45);
        return (
          <rect key={idx} x={x + 3} y={y0 + 40 - h} width={cellW - 6} height={h} fill={mode === "concat" ? headColors[Math.floor(idx / neighbors.length) % headColors.length] : "#9d174d"} opacity={0.85} />
        );
      })}
    </svg>
  );
}
