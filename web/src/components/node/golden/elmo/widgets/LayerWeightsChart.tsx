import { LAYER_WEIGHTS } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  highlightTask: string | null;
}

// 5 个任务 × 3 层的 stacked bars
export function LayerWeightsChart({ highlightTask }: Props) {
  const PAD_L = 130;
  const PAD_R = 30;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;
  const barH = 32;
  const gap = 8;

  const wOf = (v: number) => v * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Layer weights by NLP task">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        任务偏好不同层 — 底层偏语法 · 顶层偏语义
      </text>

      {/* legend */}
      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#fef3c7" stroke="#f59e0b" />
        <text x={18} y={10} fontSize={10} fill="#374151">char-CNN</text>
        <rect x={100} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={118} y={10} fontSize={10} fill="#374151">LSTM L1</text>
        <rect x={200} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={218} y={10} fontSize={10} fill="#374151">LSTM L2</text>
      </g>

      {LAYER_WEIGHTS.map((row, i) => {
        const y = PAD_T + i * (barH + gap);
        const isFocus = !highlightTask || highlightTask === row.task;
        const dim = isFocus ? 1 : 0.35;
        return (
          <g key={row.task} opacity={dim}>
            <text x={PAD_L - 8} y={y + barH / 2 + 4} textAnchor="end"
                  fontSize={11} fontWeight={isFocus ? 700 : 500} fill="#374151">
              {row.task}
            </text>

            {/* stacked bars */}
            <rect x={PAD_L} y={y} width={wOf(row.charCnn)} height={barH}
                  fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} />
            <rect x={PAD_L + wOf(row.charCnn)} y={y} width={wOf(row.lstmL1)} height={barH}
                  fill="#fce7f3" stroke="#ec4899" strokeWidth={1} />
            <rect x={PAD_L + wOf(row.charCnn) + wOf(row.lstmL1)} y={y}
                  width={wOf(row.lstmL2)} height={barH}
                  fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} />

            {/* percentages inside */}
            <text x={PAD_L + wOf(row.charCnn) / 2} y={y + barH / 2 + 4}
                  textAnchor="middle" fontSize={10} fontWeight={600} fill="#92400e">
              {(row.charCnn * 100).toFixed(0)}%
            </text>
            <text x={PAD_L + wOf(row.charCnn) + wOf(row.lstmL1) / 2} y={y + barH / 2 + 4}
                  textAnchor="middle" fontSize={10} fontWeight={600} fill="#831843">
              {(row.lstmL1 * 100).toFixed(0)}%
            </text>
            <text x={PAD_L + wOf(row.charCnn) + wOf(row.lstmL1) + wOf(row.lstmL2) / 2}
                  y={y + barH / 2 + 4}
                  textAnchor="middle" fontSize={10} fontWeight={600} fill="#1e40af">
              {(row.lstmL2 * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
