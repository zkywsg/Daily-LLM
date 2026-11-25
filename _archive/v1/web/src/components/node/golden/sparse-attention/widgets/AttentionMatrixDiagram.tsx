import { buildAttentionMatrix } from "../lib/data";

const W = 700;
const H = 380;

interface Props {
  showLocal: boolean;
  showGlobal: boolean;
  showRandom: boolean;
}

const N = 32;
const WINDOW = 6;
const GLOBAL_IDX = [0, 16];
const RANDOM_PER_ROW = 1;

const COLOR: Record<string, string> = {
  local: "#3b82f6",
  global: "#f59e0b",
  random: "#ec4899",
  none: "#f3f4f6",
};

export function AttentionMatrixDiagram({ showLocal, showGlobal, showRandom }: Props) {
  const matrix = buildAttentionMatrix(N, WINDOW, GLOBAL_IDX, RANDOM_PER_ROW);
  const size = 300;
  const cell = size / N;
  const startX = (W - size) / 2;
  const startY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Sparse attention matrix pattern">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        N×N Attention 矩阵 — Local + Global + Random 三类稀疏连接
      </text>

      <rect x={startX} y={startY} width={size} height={size} fill="none" stroke="#d1d5db" strokeWidth={1} />

      {matrix.map((row, i) =>
        row.map((kind, j) => {
          if (kind === "none") return null;
          if (kind === "local" && !showLocal) return null;
          if (kind === "global" && !showGlobal) return null;
          if (kind === "random" && !showRandom) return null;
          return (
            <rect key={`${i}-${j}`}
                  x={startX + j * cell} y={startY + i * cell}
                  width={cell} height={cell}
                  fill={COLOR[kind]} opacity={0.85} />
          );
        })
      )}

      <text x={startX + size / 2} y={startY + size + 20} textAnchor="middle" fontSize={10} fill="#6b7280">key position j →</text>
      <text x={startX - 12} y={startY + size / 2} transform={`rotate(-90, ${startX - 12}, ${startY + size / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">query position i ↓</text>

      <g transform={`translate(${startX}, ${startY + size + 40})`}>
        <rect x={0} y={0} width={12} height={12} fill="#3b82f6" opacity={0.85} />
        <text x={18} y={10} fontSize={10} fill="#374151">local(对角带)</text>
        <rect x={140} y={0} width={12} height={12} fill="#f59e0b" opacity={0.85} />
        <text x={158} y={10} fontSize={10} fill="#374151">global(十字)</text>
        <rect x={280} y={0} width={12} height={12} fill="#ec4899" opacity={0.85} />
        <text x={298} y={10} fontSize={10} fill="#374151">random(散点)</text>
      </g>

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        w + g + r ≈ 500-550 不随 N 增长 → 复杂度 O(N),而非 dense 的 O(N²)
      </text>
    </svg>
  );
}
