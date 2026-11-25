import { COMPLEXITY_TABLE } from "../lib/data";

const W = 700;
const H = 320;

export function ComplexityChart() {
  const PAD_L = 70;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const logMax = Math.log10(Math.max(...COMPLEXITY_TABLE.map((r) => r.fullAttnOps)));
  const logMin = Math.log10(Math.min(...COMPLEXITY_TABLE.map((r) => r.windowAttnOps)));

  const xOf = (i: number) => PAD_L + (i / (COMPLEXITY_TABLE.length - 1)) * plotW;
  const yOf = (v: number) => PAD_T + (1 - (Math.log10(v) - logMin) / (logMax - logMin)) * plotH;

  const fullPts = COMPLEXITY_TABLE.map((r, i) => `${xOf(i)},${yOf(r.fullAttnOps)}`).join(" ");
  const winPts = COMPLEXITY_TABLE.map((r, i) => `${xOf(i)},${yOf(r.windowAttnOps)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Full vs windowed attention complexity">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Attention 复杂度 — O(N²) vs O(N)(log 纵轴)
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {COMPLEXITY_TABLE.map((r, i) => (
        <text key={i} x={xOf(i)} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">{r.resolution}²</text>
      ))}
      <text x={W / 2} y={H - 22} textAnchor="middle" fontSize={10} fill="#6b7280">输入分辨率</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">attention 操作数(log)</text>

      <polyline points={fullPts} fill="none" stroke="#ec4899" strokeWidth={2.4} />
      <polyline points={winPts} fill="none" stroke="#10b981" strokeWidth={2.4} />

      {COMPLEXITY_TABLE.map((r, i) => (
        <g key={i}>
          <circle cx={xOf(i)} cy={yOf(r.fullAttnOps)} r={5} fill="#ec4899" stroke="#fff" strokeWidth={1.5} />
          <text x={xOf(i)} y={yOf(r.fullAttnOps) - 10} textAnchor="middle" fontSize={9} fontWeight={700} fill="#ec4899">
            {r.fullAttnOps > 1e6 ? `${(r.fullAttnOps / 1e6).toFixed(1)}M` : `${(r.fullAttnOps / 1e3).toFixed(0)}K`}
          </text>
          <circle cx={xOf(i)} cy={yOf(r.windowAttnOps)} r={5} fill="#10b981" stroke="#fff" strokeWidth={1.5} />
          <text x={xOf(i)} y={yOf(r.windowAttnOps) + 18} textAnchor="middle" fontSize={9} fontWeight={700} fill="#10b981">
            {(r.windowAttnOps / 1e3).toFixed(0)}K
          </text>
        </g>
      ))}

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#ec4899" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={10} fontWeight={600} fill="#ec4899">全局 attention O(N²)</text>
        <line x1={200} y1={6} x2={220} y2={6} stroke="#10b981" strokeWidth={2.5} />
        <text x={226} y={10} fontSize={10} fontWeight={600} fill="#10b981">windowed O(N)</text>
      </g>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        800² 时全局 attention 6.25M 次 vs windowed 仅 245K 次 — detection 分辨率不再 OOM
      </text>
    </svg>
  );
}
