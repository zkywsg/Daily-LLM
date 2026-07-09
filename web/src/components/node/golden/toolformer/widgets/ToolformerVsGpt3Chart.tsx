import { BENCHMARK_TABLE } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  highlightIdx: number;
}

export function ToolformerVsGpt3Chart({ highlightIdx }: Props) {
  const PAD_L = 150;
  const PAD_T = 50;
  const plotW = 400;
  const rowH = 46;
  const maxVal = 42;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Toolformer 6.7B vs GPT-3 175B benchmark 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        6.7B 反超 175B — 工具使用能力压缩进参数
      </text>

      <g transform={`translate(${PAD_L}, 36)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={8} fill="#374151">GPT-3 175B</text>
        <rect x={100} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={116} y={9} fontSize={8} fill="#374151">Toolformer 6.7B</text>
      </g>

      {BENCHMARK_TABLE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        return (
          <g key={row.task} opacity={isFocus ? 1 : 0.3}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={10} fontWeight={700} fill="#374151">{row.task}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.gpt3175B), 4)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.gpt3175B), 4) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.gpt3175B}</text>

            <rect x={PAD_L} y={y + 18} width={Math.max(wOf(row.toolformer), 4)} height={16} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.toolformer), 4) + 6} y={y + 31} fontSize={9} fontWeight={700} fill="#065f46">{row.toolformer}</text>
          </g>
        );
      })}
    </svg>
  );
}
