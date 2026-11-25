import { TOPK_COMPARE } from "../lib/data";

interface Props {
  selectedK: number;
}

const W = 700;
const H = 320;

// 不同 top-k 选择的质量 vs 算力对比柱+折线。
// 凸显 k=2 是 Mixtral 的 sweet spot。

export function TopKCompare({ selectedK }: Props) {
  const maxQuality = Math.max(...TOPK_COMPARE.map((t) => t.quality));
  const PAD = { left: 60, right: 40, top: 50, bottom: 90 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barW = (innerW / TOPK_COMPARE.length) * 0.55;
  const slot = innerW / TOPK_COMPARE.length;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Top-k routing quality vs compute, selected k=${selectedK}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Top-k 选择 — 质量 vs 算力权衡
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* y 刻度 (质量) */}
      {[0.8, 0.85, 0.9, 0.95].map((v) => (
        <g key={v}>
          <text x={PAD.left - 6} y={PAD.top + (1 - (v - 0.8) / 0.15) * innerH + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {(v * 100).toFixed(0)}
          </text>
          <line x1={PAD.left} x2={W - PAD.right} y1={PAD.top + (1 - (v - 0.8) / 0.15) * innerH} y2={PAD.top + (1 - (v - 0.8) / 0.15) * innerH} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      {TOPK_COMPARE.map((row, i) => {
        const x = PAD.left + i * slot + slot / 2;
        const qH = ((row.quality - 0.8) / 0.15) * innerH;
        const isCur = row.k === selectedK;
        return (
          <g key={row.k}>
            {/* 质量柱 */}
            <rect x={x - barW / 2} y={H - PAD.bottom - qH} width={barW} height={qH} rx={3} fill={isCur ? "#ec4899" : "#fce7f3"} stroke={isCur ? "#831843" : "#ec4899"} strokeWidth={isCur ? 2 : 1} />
            <text x={x} y={H - PAD.bottom - qH - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill={isCur ? "#831843" : "var(--ink-primary)"}>
              {(row.quality * 100).toFixed(1)}%
            </text>
            {/* x 标 */}
            <text x={x} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fontWeight={isCur ? 700 : 500} fill="var(--ink-primary)">
              top-{row.k}
            </text>
            <text x={x} y={H - PAD.bottom + 32} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              算力 {row.computeCostX.toFixed(1)}×
            </text>
            {/* hover/note 在底部固定区域写当前选中的注释 */}
          </g>
        );
      })}

      {/* 当前 k 的 note */}
      <rect x={20} y={H - 58} width={W - 40} height={48} rx={4} fill="#fef3c7" stroke="#f59e0b" />
      <text x={32} y={H - 38} fontSize={11} fontWeight={700} fill="#92400e">
        top-{selectedK}:
      </text>
      <text x={90} y={H - 38} fontSize={11} fill="var(--ink-primary)">
        {TOPK_COMPARE.find((r) => r.k === selectedK)?.note ?? "—"}
      </text>
    </svg>
  );
}
