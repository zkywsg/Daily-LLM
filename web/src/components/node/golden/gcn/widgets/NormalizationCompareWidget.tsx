import { rawNeighbors, rawWeight, normWeight, degree } from "../lib/data";

interface Props {
  center: number;
}

const W = 680;
const H = 320;

// 选中一个中心节点,对比它每个邻居在 "未归一化求和(权重恒为 1)"
// vs "对称归一化 D̃^(-1/2)ÃD̃^(-1/2)" 下的聚合权重差异 —— 度数越高的
// 邻居,归一化权重被压得越低,避免它在聚合里占主导。

export function NormalizationCompareWidget({ center }: Props) {
  const neighbors = rawNeighbors(center);
  const PAD = { left: 50, right: 20, top: 60, bottom: 60 };
  const innerW = W - PAD.left - PAD.right;
  const groupW = innerW / Math.max(neighbors.length, 1);
  const barW = groupW * 0.32;
  const maxH = H - PAD.top - PAD.bottom;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的邻居聚合权重对比`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        节点 {center} 的邻居聚合权重:原始求和 vs 对称归一化
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        deg({center}) = {degree(center, true)}(含自环)
      </text>

      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {neighbors.map((nb, idx) => {
        const raw = rawWeight(center, nb);
        const norm = normWeight(center, nb, true);
        const gx = PAD.left + idx * groupW + groupW / 2;
        const rawH = raw * maxH * 0.8;
        const normH = norm * maxH * 3;
        return (
          <g key={nb}>
            <rect x={gx - barW - 2} y={H - PAD.bottom - rawH} width={barW} height={rawH} fill="#9ca3af" />
            <text x={gx - barW / 2 - 2} y={H - PAD.bottom - rawH - 6} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">
              {raw.toFixed(2)}
            </text>
            <rect x={gx + 2} y={H - PAD.bottom - normH} width={barW} height={normH} fill="#ec4899" />
            <text x={gx + barW / 2 + 2} y={H - PAD.bottom - normH - 6} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">
              {norm.toFixed(2)}
            </text>
            <text x={gx} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink-secondary)">
              邻居 {nb}(deg={degree(nb, true)})
            </text>
          </g>
        );
      })}

      <g transform={`translate(${W - 170}, ${PAD.top - 30})`}>
        <rect x={0} y={0} width={12} height={12} fill="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="var(--ink-secondary)">原始求和(恒为 1)</text>
        <rect x={0} y={16} width={12} height={12} fill="#ec4899" />
        <text x={18} y={26} fontSize={10} fill="var(--ink-secondary)">对称归一化</text>
      </g>
    </svg>
  );
}
