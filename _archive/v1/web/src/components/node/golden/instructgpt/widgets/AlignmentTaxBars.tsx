import { ALIGNMENT_TAX } from "../lib/data";

const W = 700;
const H = 320;

// Alignment Tax:RLHF 对齐后 NLP benchmark 掉点,但人类偏好任务大涨。
// 横向双条:pretrain vs aligned 同组对比,差值用颜色表示。

export function AlignmentTaxBars() {
  const barH = 22;
  const rowH = 42;
  const startY = 50;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Alignment tax across tasks">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Alignment Tax — RLHF 在某些任务上付出的代价
      </text>

      {/* 图例 */}
      <g transform="translate(20, 36)">
        <rect x={0} y={-8} width={14} height={10} fill="#9ca3af" />
        <text x={18} y={2} fontSize={10} fill="var(--ink-secondary)">pretrain</text>
        <rect x={80} y={-8} width={14} height={10} fill="#ec4899" />
        <text x={98} y={2} fontSize={10} fill="var(--ink-secondary)">aligned</text>
      </g>

      {ALIGNMENT_TAX.map((row, i) => {
        const y = startY + i * rowH;
        const xBase = 200;
        const maxW = W - xBase - 80;
        const preW = row.pretrainScore * maxW;
        const alW = row.alignedScore * maxW;
        const diff = row.alignedScore - row.pretrainScore;
        const diffColor = diff >= 0 ? "#10b981" : "#dc2626";
        return (
          <g key={row.task}>
            <text x={xBase - 8} y={y + 14} textAnchor="end" fontSize={11} fontWeight={500} fill="var(--ink-primary)">
              {row.task}
            </text>
            <rect x={xBase} y={y} width={preW} height={barH / 2 - 1} fill="#9ca3af" opacity={0.75} />
            <rect x={xBase} y={y + barH / 2 + 1} width={alW} height={barH / 2 - 1} fill="#ec4899" opacity={0.85} />
            <text x={xBase + Math.max(preW, alW) + 6} y={y + 14} fontSize={11} fontWeight={700} fill={diffColor}>
              {diff >= 0 ? "+" : ""}{(diff * 100).toFixed(1)}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        通识 / 常识 / 代码 / 事实 全部小幅下降 · \"人类偏好\" 一项暴涨 · OpenAI 认为这单买卖很值
      </text>
    </svg>
  );
}
