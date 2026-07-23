import { rawNeighbors, attentionLogit, softmax } from "../lib/data";

interface Props {
  center: number;
  temperature: number;
}

const W = 680;
const H = 300;

export function AttentionBarWidget({ center, temperature }: Props) {
  const neighbors = rawNeighbors(center);
  const logits = neighbors.map((n) => attentionLogit(center, n, temperature));
  const weights = softmax(logits);

  const PAD = { left: 50, right: 20, top: 50, bottom: 50 };
  const innerW = W - PAD.left - PAD.right;
  const barW = (innerW / Math.max(neighbors.length, 1)) * 0.5;
  const gap = (innerW / Math.max(neighbors.length, 1)) * 0.5;
  const maxH = H - PAD.top - PAD.bottom;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的注意力权重`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        节点 {center} 对各邻居的 attention 权重(softmax 后)
      </text>

      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {neighbors.map((nb, idx) => {
        const w = weights[idx];
        const x = PAD.left + idx * (barW + gap) + gap / 2;
        const h = Math.min(w * maxH * 3, maxH - 10);
        return (
          <g key={nb}>
            <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} rx={3} fill="#ec4899" opacity={0.85} />
            <text x={x + barW / 2} y={H - PAD.bottom - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {w.toFixed(2)}
            </text>
            <text x={x + barW / 2} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
              邻居 {nb}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        温度越高,权重分布越集中在少数邻居上(越"尖锐")
      </text>
    </svg>
  );
}
