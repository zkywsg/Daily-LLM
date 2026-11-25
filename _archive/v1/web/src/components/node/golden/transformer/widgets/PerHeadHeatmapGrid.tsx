import { useMemo } from "react";
import { demoQKV, multiHeadWeights } from "../lib/math";

interface Props {
  tokens: string[];
  dModel: number;
  numHeads: number;
}

const CELL_SIZE = 130;
const PAD = 22;

function cellFill(v: number, max: number): string {
  if (max === 0) return "hsl(330, 80%, 90%)";
  const t = v / max;
  const light = 95 - t * 50;
  return `hsl(330, 80%, ${light}%)`;
}

// 每个 head 一张小热力图,把 d_model 切成 h 个子空间分别算 attention。
// viewer 能直观看到:不同 head 关注模式不一样 —— 有的看相邻、有的看 verb→obj。
export function PerHeadHeatmapGrid({ tokens, dModel, numHeads }: Props) {
  const heads = useMemo(() => {
    const { Q, K } = demoQKV(tokens, dModel);
    return multiHeadWeights(Q, K, numHeads, true);
  }, [tokens, dModel, numHeads]);

  const cols = Math.min(numHeads, 4);
  const rows = Math.ceil(numHeads / cols);
  const W = cols * (CELL_SIZE + PAD) + PAD;
  const H = rows * (CELL_SIZE + PAD) + PAD;
  const n = tokens.length;
  const inner = CELL_SIZE - 18;
  const cw = inner / n;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`${numHeads} 个 head 的注意力模式`}
    >
      {heads.map((mat, idx) => {
        const col = idx % cols;
        const row = Math.floor(idx / cols);
        const ox = PAD + col * (CELL_SIZE + PAD);
        const oy = PAD + row * (CELL_SIZE + PAD);
        const flat = mat.flat();
        const max = Math.max(...flat);
        return (
          <g key={idx} transform={`translate(${ox}, ${oy})`}>
            <text
              x={CELL_SIZE / 2}
              y={12}
              textAnchor="middle"
              fontSize={11}
              fontWeight={600}
              fill="var(--ink-secondary)"
            >
              head {idx + 1}
            </text>
            {mat.map((rowVals, i) =>
              rowVals.map((v, j) => (
                <rect
                  key={`${i}-${j}`}
                  x={9 + j * cw}
                  y={18 + i * cw}
                  width={cw}
                  height={cw}
                  fill={cellFill(v, max)}
                  stroke="var(--bg-canvas)"
                  strokeWidth={0.3}
                />
              )),
            )}
          </g>
        );
      })}
    </svg>
  );
}
