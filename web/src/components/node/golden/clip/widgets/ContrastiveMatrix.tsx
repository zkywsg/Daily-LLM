import { useMemo } from "react";
import { PAIRS } from "../lib/data";

interface Props {
  batchSize: number;
  /** 当前 hover 的对角线索引(用来高亮对应行 / 列) */
  hoverIdx: number | null;
  onHoverIdxChange: (i: number | null) => void;
}

const W = 700;
const H = 480;
const MARGIN = { left: 130, top: 90, right: 30, bottom: 60 };

// CLIP 训练的核心矩阵:N 张图 × N 个 caption,对角线是正样本(配对),其它都是负样本。
// 颜色用 hsl(330, ...) 粉调,对角线高亮(深粉),其余浅。
// hover 行/列时把对应的图/文 label 高亮,让 viewer 看到"contrastive 在惩罚谁、奖励谁"。

function simulateScore(imgIdx: number, txtIdx: number, batchSize: number): number {
  // 假装是真模型的输出:对角线高,同 group 中等,跨 group 低。
  if (imgIdx >= batchSize || txtIdx >= batchSize) return 0;
  if (imgIdx === txtIdx) return 0.85 + 0.1 * ((imgIdx * 7) % 10) / 10;
  const sameGroup = PAIRS[imgIdx]?.group === PAIRS[txtIdx]?.group;
  if (sameGroup) return 0.35 + 0.15 * ((imgIdx + txtIdx) % 7) / 7;
  return 0.08 + 0.1 * ((imgIdx + 3 * txtIdx) % 5) / 5;
}

export function ContrastiveMatrix({ batchSize, hoverIdx, onHoverIdxChange }: Props) {
  const N = Math.min(batchSize, PAIRS.length);
  const innerW = W - MARGIN.left - MARGIN.right;
  const innerH = H - MARGIN.top - MARGIN.bottom;
  const cellSize = Math.min(innerW, innerH) / N;
  const matSize = cellSize * N;

  const scores = useMemo(() => {
    const rows: number[][] = [];
    for (let i = 0; i < N; i++) {
      const row: number[] = [];
      for (let j = 0; j < N; j++) row.push(simulateScore(i, j, N));
      rows.push(row);
    }
    return rows;
  }, [N]);

  const cellFill = (v: number): string => {
    const light = 95 - v * 55;
    return `hsl(330, 70%, ${light}%)`;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Contrastive similarity matrix, batch ${N}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        N×N 相似度矩阵 — 对角线(配对)拉高,其它(非配对)压低
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        鼠标 hover 行 / 列看 contrastive 在惩罚哪些负样本
      </text>

      {/* 列标题:caption 缩写 */}
      {Array.from({ length: N }, (_, j) => (
        <text
          key={`c-${j}`}
          x={MARGIN.left + (j + 0.5) * cellSize}
          y={MARGIN.top - 12}
          textAnchor="middle"
          fontSize={10}
          fill={hoverIdx === j ? "#ec4899" : "var(--ink-secondary)"}
          fontWeight={hoverIdx === j ? 700 : 400}
        >
          "{PAIRS[j].caption.split(" ").pop()}"
        </text>
      ))}
      <text x={MARGIN.left + matSize / 2} y={MARGIN.top - 26} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        text tokens →
      </text>

      {/* 行标题:image emoji */}
      {Array.from({ length: N }, (_, i) => (
        <g key={`r-${i}`}>
          <text
            x={MARGIN.left - 12}
            y={MARGIN.top + (i + 0.5) * cellSize + 6}
            textAnchor="end"
            fontSize={18}
          >
            {PAIRS[i].emoji}
          </text>
          {hoverIdx === i && (
            <circle cx={MARGIN.left - 38} cy={MARGIN.top + (i + 0.5) * cellSize} r={4} fill="#ec4899" />
          )}
        </g>
      ))}
      <text x={MARGIN.left - 60} y={MARGIN.top + matSize / 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)" transform={`rotate(-90 ${MARGIN.left - 60} ${MARGIN.top + matSize / 2})`}>
        ↓ images
      </text>

      {/* cells */}
      {scores.map((row, i) =>
        row.map((v, j) => {
          const isDiag = i === j;
          const inHover = hoverIdx != null && (hoverIdx === i || hoverIdx === j);
          return (
            <g
              key={`${i}-${j}`}
              onMouseEnter={() => onHoverIdxChange(i)}
              onMouseLeave={() => onHoverIdxChange(null)}
              style={{ cursor: "default" }}
            >
              <rect
                x={MARGIN.left + j * cellSize}
                y={MARGIN.top + i * cellSize}
                width={cellSize}
                height={cellSize}
                fill={cellFill(v)}
                stroke={isDiag ? "#ec4899" : "var(--bg-canvas)"}
                strokeWidth={isDiag ? 2 : 0.5}
                opacity={hoverIdx == null || inHover ? 1 : 0.35}
              />
              <text
                x={MARGIN.left + (j + 0.5) * cellSize}
                y={MARGIN.top + (i + 0.5) * cellSize + 4}
                textAnchor="middle"
                fontSize={10}
                fontWeight={isDiag ? 700 : 400}
                fill={v > 0.5 ? "#fff" : "var(--ink-primary)"}
                style={{ pointerEvents: "none" }}
              >
                {v.toFixed(2)}
              </text>
            </g>
          );
        }),
      )}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#10b981">
        ✓ 对角线 N 个正样本 / · N²−N 个负样本(同 batch 内随机配)
      </text>
    </svg>
  );
}
