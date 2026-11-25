import { LABEL_PIPELINE_COMPARE } from "../lib/data";

const W = 700;
const H = 300;

export function RlaifVsRlhfDiagram() {
  const PAD_L = 40;
  const PAD_R = 40;
  const colW = (W - PAD_L - PAD_R - 40) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RLAIF vs RLHF 标注流程对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        人工标注流程 vs AI 标注流程 — 同样产出约 30K 偏好对
      </text>

      {/* 人工 RLHF 流程 */}
      <g transform={`translate(${PAD_L}, 45)`}>
        <text x={colW / 2} y={0} textAnchor="middle" fontSize={11} fontWeight={700} fill="#3b82f6">
          InstructGPT(人工 RLHF)
        </text>
        <rect x={0} y={15} width={colW} height={36} fill="#dbeafe" stroke="#3b82f6" rx={4} />
        <text x={colW / 2} y={37} textAnchor="middle" fontSize={9} fill="#1e40af">40 名标注员 · 6 个月</text>

        <path d={`M ${colW / 2} 51 L ${colW / 2} 70`} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arr-a)" />

        <rect x={0} y={70} width={colW} height={36} fill="#dbeafe" stroke="#3b82f6" rx={4} />
        <text x={colW / 2} y={92} textAnchor="middle" fontSize={9} fill="#1e40af">人工读 + 排序候选回答</text>

        <path d={`M ${colW / 2} 106 L ${colW / 2} 125`} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arr-a)" />

        <rect x={0} y={125} width={colW} height={36} fill="#fce7f3" stroke="#ec4899" rx={4} />
        <text x={colW / 2} y={147} textAnchor="middle" fontSize={9} fontWeight={700} fill="#9d174d">33K 人类偏好对</text>
        <text x={colW / 2} y={178} textAnchor="middle" fontSize={9} fill="#ef4444">⚠ 有害内容标注对心理伤害大</text>
      </g>

      {/* RLAIF 流程 */}
      <g transform={`translate(${PAD_L + colW + 40}, 45)`}>
        <text x={colW / 2} y={0} textAnchor="middle" fontSize={11} fontWeight={700} fill="#10b981">
          Constitutional AI(RLAIF)
        </text>
        <rect x={0} y={15} width={colW} height={36} fill="#ecfdf5" stroke="#10b981" rx={4} />
        <text x={colW / 2} y={37} textAnchor="middle" fontSize={9} fill="#065f46">16 条 constitution(写一次)</text>

        <path d={`M ${colW / 2} 51 L ${colW / 2} 70`} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arr-a)" />

        <rect x={0} y={70} width={colW} height={36} fill="#ecfdf5" stroke="#10b981" rx={4} />
        <text x={colW / 2} y={92} textAnchor="middle" fontSize={9} fill="#065f46">AI 按 constitution 排序候选</text>

        <path d={`M ${colW / 2} 106 L ${colW / 2} 125`} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arr-a)" />

        <rect x={0} y={125} width={colW} height={36} fill="#fef3c7" stroke="#f59e0b" rx={4} />
        <text x={colW / 2} y={147} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">~30K AI 偏好对</text>
        <text x={colW / 2} y={178} textAnchor="middle" fontSize={9} fill="#10b981">✓ 无疲劳 · 无心理伤害 · 可无限扩展</text>
      </g>

      <defs>
        <marker id="arr-a" markerWidth={8} markerHeight={8} refX={6} refY={3} orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>

      {LABEL_PIPELINE_COMPARE.map((row, i) => (
        <text key={row.method} x={W / 2} y={215 + i * 16} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
          {row.method}:{row.laborMonths > 0 ? `${row.laborMonths} 人月` : "0 人月(纯算力)"} · ~${(row.costUSD / 1_000_000).toFixed(2)}M · {row.pairs}K 偏好对
        </text>
      ))}

      <text x={W / 2} y={215 + LABEL_PIPELINE_COMPARE.length * 16 + 16} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        对齐成本从"人力密集"压到"算力密集" — 约降低一个数量级
      </text>
    </svg>
  );
}
