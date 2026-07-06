import { AGENT_TASK_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

export function AgentTaskCompareChart() {
  const PAD_L = 150;
  const PAD_T = 40;
  const plotW = 420;
  const rowH = 70;
  const maxVal = 80;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ALFWorld / WebShop agent 任务成功率对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        不微调,纯 prompt 就把模仿学习基线打掉一截
      </text>

      <g transform={`translate(${PAD_L}, 32)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={8} fill="#374151">Standard</text>
        <rect x={80} y={0} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={96} y={9} fontSize={8} fill="#374151">Imitation Learning</text>
        <rect x={220} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={236} y={9} fontSize={8} fill="#374151">ReAct</text>
      </g>

      {AGENT_TASK_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        return (
          <g key={row.task}>
            <text x={PAD_L - 10} y={y + 24} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.task}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.standard), 4)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.standard), 4) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.standard}%</text>

            <rect x={PAD_L} y={y + 18} width={Math.max(wOf(row.imitationLearning), 4)} height={16} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.imitationLearning), 4) + 6} y={y + 31} fontSize={9} fill="#1e40af">{row.imitationLearning}%</text>

            <rect x={PAD_L} y={y + 36} width={Math.max(wOf(row.react), 4)} height={16} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.react), 4) + 6} y={y + 49} fontSize={9} fontWeight={700} fill="#065f46">{row.react}%</text>
          </g>
        );
      })}
    </svg>
  );
}
