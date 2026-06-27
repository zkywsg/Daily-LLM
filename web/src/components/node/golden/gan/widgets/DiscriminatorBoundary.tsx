import { useMemo } from "react";
import { sampleReal, sampleGen, discriminate, type Point } from "../lib/dist";

interface Props {
  iter: number;
  totalIter: number;
}

const W = 700;
const H = 400;
const PAD = 36;
const GRID = 40; // 决策边界网格分辨率

// 在 2D 平面上铺一张网格,每格画 D(x) 的概率背景色;再叠加真/假样本点。
// 让 viewer 看到 D 学到的"哪里是真分布"边界。

export function DiscriminatorBoundary({ iter, totalIter }: Props) {
  const realPts = useMemo(() => sampleReal(80, 42), []);
  const fakePts = useMemo(() => sampleGen(80, iter, totalIter, 7), [iter, totalIter]);

  const xScale = (x: number) => PAD + ((x + 3) / 6) * (W - 2 * PAD);
  const yScale = (y: number) => PAD + ((y + 3) / 6) * (H - 2 * PAD);

  // 网格 cell 大小
  const cellW = (W - 2 * PAD) / GRID;
  const cellH = (H - 2 * PAD) / GRID;

  // 生成网格背景
  const grid: Array<{ x: number; y: number; score: number }> = [];
  for (let gx = 0; gx < GRID; gx++) {
    for (let gy = 0; gy < GRID; gy++) {
      const dx = -3 + 6 * (gx / GRID);
      const dy = -3 + 6 * (gy / GRID);
      const p: Point = { x: dx, y: dy };
      grid.push({ x: gx, y: gy, score: discriminate(p, iter, totalIter) });
    }
  }

  // 概率背景色:1=粉(真),0=蓝(假),0.5=灰
  const cellFill = (s: number) => {
    if (s > 0.5) {
      // 偏真 → 粉
      const t = (s - 0.5) * 2;
      return `hsl(330, 60%, ${95 - t * 30}%)`;
    } else {
      // 偏假 → 蓝
      const t = (0.5 - s) * 2;
      return `hsl(220, 60%, ${95 - t * 30}%)`;
    }
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Discriminator decision boundary, iter ${iter}/${totalIter}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        D 决策边界 — iter {iter} / {totalIter} (粉=判真 / 蓝=判假 / 灰=不确定)
      </text>

      {/* 背景网格 */}
      {grid.map((g, i) => (
        <rect
          key={i}
          x={PAD + g.x * cellW}
          y={PAD + g.y * cellH}
          width={cellW + 0.5}
          height={cellH + 0.5}
          fill={cellFill(g.score)}
          opacity={0.55}
        />
      ))}

      {/* 真样本 */}
      {realPts.map((p, i) => (
        <circle
          key={`r-${i}`}
          cx={xScale(p.x)}
          cy={yScale(p.y)}
          r={3}
          fill="#831843"
          stroke="#fff"
          strokeWidth={1}
        />
      ))}

      {/* 假样本 */}
      {fakePts.map((p, i) => (
        <circle
          key={`f-${i}`}
          cx={xScale(p.x)}
          cy={yScale(p.y)}
          r={3}
          fill="#1e3a8a"
          stroke="#fff"
          strokeWidth={1}
        />
      ))}

      {/* 图例 */}
      <g transform={`translate(${PAD + 10}, ${H - PAD - 28})`}>
        <circle cx={6} cy={6} r={4} fill="#831843" stroke="#fff" strokeWidth={1} />
        <text x={16} y={10} fontSize={11} fill="var(--ink-secondary)">真样本</text>
        <g transform="translate(80, 0)">
          <circle cx={6} cy={6} r={4} fill="#1e3a8a" stroke="#fff" strokeWidth={1} />
          <text x={16} y={10} fontSize={11} fill="var(--ink-secondary)">G 假样本</text>
        </g>
      </g>
    </svg>
  );
}
