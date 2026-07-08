import { BRIDGE_PARAM_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function BridgeParamCompareChart() {
  const PAD_L = 220;
  const PAD_T = 40;
  const plotW = 380;
  const rowH = 50;
  const maxLog = Math.log10(500);
  const wOf = (v: number) => (Math.log10(v) / maxLog) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="桥接模块参数量对比(log 尺度)">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        桥接模块参数量(log 尺度)— 简洁有效胜过复杂精巧
      </text>

      {BRIDGE_PARAM_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isLlava = row.method.startsWith("LLaVA");
        const color = isLlava ? "#10b981" : "#9ca3af";
        const bg = isLlava ? "#ecfdf5" : "#f3f4f6";
        return (
          <g key={row.method}>
            <text x={PAD_L - 10} y={y + 15} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>{row.method}</text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.paramsM), 4)} height={20} fill={bg} stroke={color} strokeWidth={1.6} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.paramsM), 4) + 8} y={y + 15} fontSize={10} fontWeight={700} fill={color}>{row.paramsM}M</text>
          </g>
        );
      })}
    </svg>
  );
}
