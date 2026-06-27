import { useMemo } from "react";
import { TOY_IMAGE, simulateClsAttention } from "../lib/data";

interface Props {
  patchSize: number;
}

const W = 700;
const H = 320;

// 把 CLS 对每个 patch 的 attention 强度叠到原 image 上,看 CLS 关注了哪里。
// 模拟得粗糙(论文里这种图是真训完取最后一层 attention),但能体现"中心 object 高、边角低"的直觉。
export function ClsAttentionHeatmap({ patchSize }: Props) {
  const attMap = useMemo(() => simulateClsAttention(patchSize), [patchSize]);
  const N = TOY_IMAGE.length;
  const numAxis = N / patchSize;

  const leftSize = 200;
  const leftPadX = 60;
  const rightStartX = leftPadX + leftSize + 80;
  const pixSize = leftSize / N;
  const patchPixW = patchSize * pixSize;

  const imageCellFill = (v: number) => `hsl(0, 0%, ${v * 100}%)`;
  const attCellFill = (v: number) => {
    const light = 95 - v * 50;
    return `hsl(330, 70%, ${light}%)`;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CLS attention heatmap over image patches">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        CLS Token 对各 patch 的 attention 分布(全局聚合的可视化证据)
      </text>

      {/* 左:原 image */}
      <g transform={`translate(${leftPadX}, 60)`}>
        <text x={leftSize / 2} y={-8} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">原 image</text>
        {TOY_IMAGE.map((row, r) =>
          row.map((v, c) => (
            <rect key={`im-${r}-${c}`} x={c * pixSize} y={r * pixSize} width={pixSize} height={pixSize} fill={imageCellFill(v)} />
          ))
        )}
        {/* patch 网格线 */}
        {Array.from({ length: numAxis + 1 }, (_, i) => (
          <g key={`grid-${i}`}>
            <line x1={i * patchPixW} y1={0} x2={i * patchPixW} y2={leftSize} stroke="#6b7280" strokeWidth={0.4} />
            <line x1={0} y1={i * patchPixW} x2={leftSize} y2={i * patchPixW} stroke="#6b7280" strokeWidth={0.4} />
          </g>
        ))}
      </g>

      {/* 中间箭头 */}
      <text x={leftPadX + leftSize + 40} y={H / 2 + 30} textAnchor="middle" fontSize={22} fill="var(--ink-muted)">→</text>
      <text x={leftPadX + leftSize + 40} y={H / 2 + 50} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        CLS att
      </text>

      {/* 右:attention map */}
      <g transform={`translate(${rightStartX}, 60)`}>
        <text x={leftSize / 2} y={-8} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">CLS attention map</text>
        {attMap.map((row, r) =>
          row.map((v, c) => (
            <g key={`a-${r}-${c}`}>
              <rect x={c * patchPixW} y={r * patchPixW} width={patchPixW} height={patchPixW} fill={attCellFill(v)} stroke="#6b7280" strokeWidth={0.4} />
              <text x={c * patchPixW + patchPixW / 2} y={r * patchPixW + patchPixW / 2 + 4} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
                {v.toFixed(2)}
              </text>
            </g>
          ))
        )}
      </g>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        粉色越深 = CLS 越关注该 patch · 中心 object 区域 attention 高 · 边角背景低
      </text>
    </svg>
  );
}
