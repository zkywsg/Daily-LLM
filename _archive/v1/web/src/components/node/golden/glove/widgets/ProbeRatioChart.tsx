import { PROBE_TABLE } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightIdx: number;
}

export function ProbeRatioChart({ highlightIdx }: Props) {
  const PAD_L = 100;
  const PAD_R = 60;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;

  const logMax = 1.2, logMin = -1.2; // log10 范围覆盖 0.06..15
  const xOf = (ratio: number) => {
    const logR = Math.log10(ratio);
    return PAD_L + ((logR - logMin) / (logMax - logMin)) * plotW;
  };
  const zeroX = xOf(1);

  const rowH = 50;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ice/steam co-occurrence probability ratio">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        P(k|ice) / P(k|steam) — 比值才是语义区分的载体
      </text>

      <line x1={zeroX} y1={PAD_T - 10} x2={zeroX} y2={PAD_T + PROBE_TABLE.length * rowH} stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="3 3" />
      <text x={zeroX} y={PAD_T - 16} textAnchor="middle" fontSize={9} fill="#6b7280">ratio=1(中性)</text>

      {PROBE_TABLE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        const color = row.ratio > 2 ? "#3b82f6" : row.ratio < 0.5 ? "#ec4899" : "#9ca3af";
        const x = xOf(row.ratio);
        return (
          <g key={row.probe} opacity={isFocus ? 1 : 0.3}>
            <text x={PAD_L - 12} y={y + 20} textAnchor="end" fontSize={12} fontWeight={700} fill="#374151">{row.probe}</text>
            <line x1={zeroX} y1={y + 16} x2={x} y2={y + 16} stroke={color} strokeWidth={2.5} />
            <circle cx={x} cy={y + 16} r={6} fill={color} stroke="#fff" strokeWidth={1.5} />
            <text x={x} y={y + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill={color}>{row.ratio.toFixed(2)}</text>
            <text x={x + (row.ratio > 1 ? 14 : -14)} y={y + 20} textAnchor={row.ratio > 1 ? "start" : "end"} fontSize={9} fill="#6b7280">{row.interp}</text>
          </g>
        );
      })}

      <text x={xOf(0.1)} y={PAD_T + PROBE_TABLE.length * rowH + 8} textAnchor="middle" fontSize={9} fill="#9ca3af">← 偏 steam</text>
      <text x={xOf(8)} y={PAD_T + PROBE_TABLE.length * rowH + 8} textAnchor="middle" fontSize={9} fill="#9ca3af">偏 ice →</text>
    </svg>
  );
}
