import { LLAVA_BENCH } from "../lib/data";

const W = 700;
const H = 320;

export function LlavaBenchChart() {
  const PAD_L = 150;
  const PAD_T = 50;
  const plotW = 420;
  const rowH = 60;
  const maxVal = 100;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LLaVA-Bench 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LLaVA-Bench(GPT-4 作 judge)— reasoning 任务接近纯文本 GPT-4
      </text>

      <g transform={`translate(${PAD_L}, 36)`}>
        <rect x={0} y={0} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={16} y={9} fontSize={8} fill="#374151">Conversation</text>
        <rect x={95} y={0} width={10} height={10} fill="#fce7f3" stroke="#ec4899" />
        <text x={111} y={9} fontSize={8} fill="#374151">Detail</text>
        <rect x={165} y={0} width={10} height={10} fill="#fef3c7" stroke="#f59e0b" />
        <text x={181} y={9} fontSize={8} fill="#374151">Reasoning</text>
        <rect x={255} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={271} y={9} fontSize={8} fill="#374151">Overall</text>
      </g>

      {LLAVA_BENCH.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isLlava = row.model === "LLaVA";
        return (
          <g key={row.model}>
            <text x={PAD_L - 10} y={y + 24} textAnchor="end" fontSize={10} fontWeight={700} fill={isLlava ? "#065f46" : "#374151"}>{row.model}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.conversation), 4)} height={10} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} rx={2} />
            <rect x={PAD_L} y={y + 12} width={Math.max(wOf(row.detail), 4)} height={10} fill="#fce7f3" stroke="#ec4899" strokeWidth={1} rx={2} />
            <rect x={PAD_L} y={y + 24} width={Math.max(wOf(row.reasoning), 4)} height={10} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} rx={2} />
            <rect x={PAD_L} y={y + 36} width={Math.max(wOf(row.overall), 4)} height={12} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.overall), 4) + 6} y={y + 46} fontSize={9} fontWeight={700} fill="#065f46">{row.overall}</text>
          </g>
        );
      })}
    </svg>
  );
}
