import { HELPFUL_HARMLESS_COMPARE } from "../lib/data";

const W = 700;
const H = 360;

export function HelpfulHarmlessTradeoffChart() {
  const PAD_L = 70;
  const PAD_R = 40;
  const PAD_T = 40;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // x = helpfulness 胜率(0-100), y = harmlessness 胜率(-30 到 +15)
  const xMin = 0;
  const xMax = 100;
  const yMin = -30;
  const yMax = 15;

  const xOf = (v: number) => PAD_L + ((v - xMin) / (xMax - xMin)) * plotW;
  const yOf = (v: number) => PAD_T + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const colorOf = (method: string) =>
    method.startsWith("Constitutional") ? "#10b981" : method.startsWith("Helpful-only") ? "#ec4899" : "#3b82f6";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="helpfulness vs harmlessness 权衡对比(论文 Table 1)">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Helpfulness vs Harmlessness 权衡(人工评分胜率,论文 Table 1)
      </text>

      {/* 坐标轴 */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <text x={PAD_L + plotW / 2} y={H - 14} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        Helpfulness 胜率 %(vs SFT-only)
      </text>
      <text x={18} y={PAD_T + plotH / 2} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" transform={`rotate(-90, 18, ${PAD_T + plotH / 2})`}>
        Harmlessness 胜率 %(vs 基准)
      </text>

      {/* 0 基准线 */}
      <line x1={PAD_L} y1={yOf(0)} x2={PAD_L + plotW} y2={yOf(0)} stroke="#9ca3af" strokeWidth={1} strokeDasharray="4 2" />
      <text x={PAD_L + plotW} y={yOf(0) - 4} textAnchor="end" fontSize={9} fill="#9ca3af">Standard RLHF 基准(0%)</text>

      {/* "评的分" 象限提示:右上为理想区(helpful 且 harmless 都高) */}
      <rect x={xOf(50)} y={PAD_T} width={PAD_L + plotW - xOf(50)} height={yOf(0) - PAD_T} fill="#ecfdf5" opacity={0.35} />
      <text x={PAD_L + plotW - 6} y={PAD_T + 14} textAnchor="end" fontSize={9} fontStyle="italic" fill="#065f46">理想区:helpful 且 harmless 都不掉</text>

      {HELPFUL_HARMLESS_COMPARE.map((row) => {
        const x = xOf(row.helpfulness);
        const y = yOf(row.harmlessness);
        const color = colorOf(row.method);
        return (
          <g key={row.method}>
            <circle cx={x} cy={y} r={7} fill={color} opacity={0.85} />
            <text x={x} y={y - 14} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>
              {row.method}
            </text>
            <text x={x} y={y + 24} textAnchor="middle" fontSize={9} fill={color}>
              ({row.helpfulness}%, {row.harmlessness > 0 ? "+" : ""}{row.harmlessness}%)
            </text>
          </g>
        );
      })}

      <text x={PAD_L} y={PAD_T + plotH + 40} fontSize={10} fill="var(--ink-muted)">
        Helpful-only baseline:helpfulness 持平但 harmlessness 大幅变差(-23%,回避/教唆风险高)。
      </text>
      <text x={PAD_L} y={PAD_T + plotH + 40 + 14} fontSize={10} fill="var(--ink-muted)">
        Constitutional AI:helpfulness 与人类 RLHF 持平(51% vs 50%),harmlessness 反超 +9% —— 没有"变得回避/不帮忙"的失败模式。
      </text>
    </svg>
  );
}
