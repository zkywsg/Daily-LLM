import { useMemo } from "react";
import { sampleReal, sampleGen, discriminate } from "../lib/dist";

interface Props {
  iter: number;
  totalIter: number;
}

const W = 700;
const H = 280;

// 8 个真样本 + 8 个 G 假样本,D 给每个打分。
// 真样本期望 D≈1,假样本期望 D≈0;训练初期所有点都≈0.5(D 还分不清)。
// 让 viewer 看到训练让 D 把两堆分数拉开。

export function DiscriminatorScoreBars({ iter, totalIter }: Props) {
  const samples = useMemo(() => {
    const real = sampleReal(8, 42);
    const fake = sampleGen(8, iter, totalIter, 7);
    return [
      ...real.map((p) => ({ p, kind: "real" as const, score: discriminate(p, iter, totalIter) })),
      ...fake.map((p) => ({ p, kind: "fake" as const, score: discriminate(p, iter, totalIter) })),
    ];
  }, [iter, totalIter]);

  const W_BAR = 22;
  const GAP = 6;
  const totalBarsW = samples.length * (W_BAR + GAP);
  const startX = (W - totalBarsW) / 2;
  const BAR_BASE_Y = H - 60;
  const BAR_MAX_H = H - 120;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Discriminator scores on real vs fake samples, iter ${iter}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        D 对 8 真 + 8 假 样本的输出 — iter {iter} / {totalIter}
      </text>

      {/* 0.5 参考线 */}
      <line
        x1={startX - 10}
        x2={startX + totalBarsW}
        y1={BAR_BASE_Y - BAR_MAX_H * 0.5}
        y2={BAR_BASE_Y - BAR_MAX_H * 0.5}
        stroke="var(--ink-muted)"
        strokeDasharray="3 3"
      />
      <text x={startX - 14} y={BAR_BASE_Y - BAR_MAX_H * 0.5 + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
        0.5
      </text>

      {/* 1.0 参考线 */}
      <line
        x1={startX - 10}
        x2={startX + totalBarsW}
        y1={BAR_BASE_Y - BAR_MAX_H}
        y2={BAR_BASE_Y - BAR_MAX_H}
        stroke="var(--ink-muted)"
        strokeDasharray="2 4"
      />
      <text x={startX - 14} y={BAR_BASE_Y - BAR_MAX_H + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
        1.0
      </text>

      {/* x 轴 */}
      <line x1={startX - 10} x2={startX + totalBarsW} y1={BAR_BASE_Y} y2={BAR_BASE_Y} stroke="var(--ink-muted)" />

      {/* 柱子 */}
      {samples.map((s, i) => {
        const x = startX + i * (W_BAR + GAP);
        const h = BAR_MAX_H * s.score;
        const color = s.kind === "real" ? "#ec4899" : "#3b82f6";
        return (
          <g key={i}>
            <rect x={x} y={BAR_BASE_Y - h} width={W_BAR} height={h} fill={color} opacity={0.8} rx={2} />
            <text x={x + W_BAR / 2} y={BAR_BASE_Y - h - 4} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
              {s.score.toFixed(2)}
            </text>
            <text x={x + W_BAR / 2} y={BAR_BASE_Y + 14} textAnchor="middle" fontSize={9} fill={color}>
              {s.kind === "real" ? "R" : "F"}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        iter=0 时 D 都给 ≈0.5(分不清)· iter→T 时 R 升到 ~1、F 降到 ~0(D 学会了)
      </text>
    </svg>
  );
}
