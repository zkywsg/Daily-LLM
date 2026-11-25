import { ALIGNMENT_DEMO, buildAlignmentMatrix } from "../lib/data";

const W = 700;
const H = 420;

interface Props {
  highlightTgt: number | null; // null = 全部高亮均匀;数字 = 高亮该目标词的 α 分布
}

const M = buildAlignmentMatrix();

export function AlignmentHeatmap({ highlightTgt }: Props) {
  const { src, tgt } = ALIGNMENT_DEMO;
  const PAD_L = 110;
  const PAD_T = 90;
  const cellW = (W - PAD_L - 30) / src.length;
  const cellH = (H - PAD_T - 20) / tgt.length;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Alignment heatmap EN to FR">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Soft Alignment — EN → FR · 注意 European/Economic/Area ↔ zone/économique/européenne 反序
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        每行 = 目标词 t 的 α_t 分布 · 颜色深浅 = 注意力权重
      </text>

      {/* src labels (top) */}
      {src.map((s, i) => (
        <text key={i} x={PAD_L + i * cellW + cellW / 2} y={PAD_T - 8}
              textAnchor="end" fontSize={10} fontWeight={500} fill="#374151"
              transform={`rotate(-45, ${PAD_L + i * cellW + cellW / 2}, ${PAD_T - 8})`}>
          {s}
        </text>
      ))}

      {/* tgt labels (left) */}
      {tgt.map((t, i) => (
        <text key={i} x={PAD_L - 8} y={PAD_T + i * cellH + cellH / 2 + 4}
              textAnchor="end" fontSize={10}
              fontWeight={highlightTgt === i ? 700 : 500}
              fill={highlightTgt === i ? "#ec4899" : "#374151"}>
          {t}
        </text>
      ))}

      {/* cells */}
      {M.map((row, t) => (
        row.map((alpha, s) => {
          const isHigh = highlightTgt === null || highlightTgt === t;
          const op = isHigh ? alpha : alpha * 0.25;
          // intensity colored
          const blue = 235 - Math.floor(alpha * 200);
          const fill = `rgb(${Math.max(blue, 30)}, ${Math.max(blue, 50)}, ${Math.min(blue + 30, 255)})`;
          return (
            <g key={`${t}-${s}`}>
              <rect x={PAD_L + s * cellW + 1} y={PAD_T + t * cellH + 1}
                    width={cellW - 2} height={cellH - 2}
                    fill={fill} opacity={op} rx={1.5} />
              {alpha > 0.3 && isHigh && (
                <text x={PAD_L + s * cellW + cellW / 2} y={PAD_T + t * cellH + cellH / 2 + 3}
                      textAnchor="middle" fontSize={8} fontWeight={700} fill="#fff">
                  {alpha.toFixed(2)}
                </text>
              )}
            </g>
          );
        })
      ))}

      {/* highlight row outline */}
      {highlightTgt !== null && (
        <rect x={PAD_L} y={PAD_T + highlightTgt * cellH}
              width={src.length * cellW} height={cellH}
              fill="none" stroke="#ec4899" strokeWidth={2} rx={2} />
      )}
    </svg>
  );
}
