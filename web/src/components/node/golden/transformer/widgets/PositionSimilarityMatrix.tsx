import { useMemo } from "react";
import { positionalEncoding, cosineSim } from "../lib/math";

interface Props {
  nPos: number;
  dModel: number;
}

const SIZE = 360;
const PAD = 36;

// PE_i · PE_j 的相似度热力图 —— viewer 看 PE 是不是真在编码"距离":
// 相邻 pos 余弦应该接近 1,距离越远越往 0 走。这是 PE 工作的直接证据。
export function PositionSimilarityMatrix({ nPos, dModel }: Props) {
  const sim = useMemo(() => {
    const PE = positionalEncoding(nPos, dModel);
    return PE.map((a) => PE.map((b) => cosineSim(a, b)));
  }, [nPos, dModel]);
  const cell = (SIZE - PAD) / nPos;
  return (
    <svg
      viewBox={`0 0 ${SIZE + 20} ${SIZE + 30}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="位置编码自相似度矩阵"
    >
      <text
        x={SIZE / 2 + 10}
        y={18}
        textAnchor="middle"
        fontSize={12}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        cos_sim(PE_i, PE_j) — 对角线 = 1
      </text>
      {sim.map((row, i) =>
        row.map((v, j) => {
          // v ∈ [-1, 1],映射到 hue:正向粉色,负向蓝灰
          const t = (v + 1) / 2;
          const hue = v >= 0 ? 330 : 220;
          const light = 95 - Math.abs(v) * 50;
          return (
            <rect
              key={`${i}-${j}`}
              x={PAD + j * cell}
              y={PAD + i * cell}
              width={cell}
              height={cell}
              fill={`hsl(${hue}, 70%, ${light}%)`}
              opacity={0.6 + t * 0.4}
              stroke="var(--bg-canvas)"
              strokeWidth={0.4}
            />
          );
        }),
      )}
      {/* 轴标 */}
      {Array.from({ length: nPos }, (_, k) => k).map((k) => (
        <g key={`lab-${k}`}>
          <text
            x={PAD + (k + 0.5) * cell}
            y={PAD - 4}
            textAnchor="middle"
            fontSize={9}
            fill="var(--ink-muted)"
          >
            {k}
          </text>
          <text
            x={PAD - 4}
            y={PAD + (k + 0.5) * cell + 3}
            textAnchor="end"
            fontSize={9}
            fill="var(--ink-muted)"
          >
            {k}
          </text>
        </g>
      ))}
    </svg>
  );
}
