import { CONTEXT_GROWTH } from "../lib/data";

const W = 700;
const H = 320;

export function EffectiveContextChart() {
  const PAD_L = 70;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const maxSeg = CONTEXT_GROWTH[CONTEXT_GROWTH.length - 1].segments;
  const maxCtx = Math.max(...CONTEXT_GROWTH.map((d) => d.transformerXL));

  const xOf = (seg: number) => PAD_L + (seg / maxSeg) * plotW;
  const yOf = (ctx: number) => PAD_T + plotH - (ctx / maxCtx) * plotH;

  const fixedPath = CONTEXT_GROWTH.map((d, i) => `${i === 0 ? "M" : "L"} ${xOf(d.segments)} ${yOf(d.fixedWindow)}`).join(" ");
  const xlPath = CONTEXT_GROWTH.map((d, i) => `${i === 0 ? "M" : "L"} ${xOf(d.segments)} ${yOf(d.transformerXL)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="有效上下文长度 vs 段数">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        有效上下文随段数增长 — 固定窗口(灰)恒为 384,Transformer-XL(蓝)随段数线性累积
      </text>

      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" strokeWidth={1.2} />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" strokeWidth={1.2} />

      {[0, 1000, 2000, 3000, 4000].map((v) => (
        <g key={v}>
          <text x={PAD_L - 8} y={yOf(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">{v}</text>
          <line x1={PAD_L} y1={yOf(v)} x2={W - PAD_R} y2={yOf(v)} stroke="#f3f4f6" strokeWidth={1} />
        </g>
      ))}
      {CONTEXT_GROWTH.map((d) => (
        <text key={d.segments} x={xOf(d.segments)} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
          {d.segments}
        </text>
      ))}
      <text x={W / 2} y={PAD_T + plotH + 36} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">段数(每段 384 token)</text>

      <path d={fixedPath} fill="none" stroke="#9ca3af" strokeWidth={1.6} strokeDasharray="5 3" />
      <path d={xlPath} fill="none" stroke="#3b82f6" strokeWidth={2} />

      <circle cx={xOf(10)} cy={yOf(CONTEXT_GROWTH[10].transformerXL)} r={4} fill="#3b82f6" />
      <text x={xOf(10) - 8} y={yOf(CONTEXT_GROWTH[10].transformerXL) - 10} textAnchor="end" fontSize={10} fontWeight={700} fill="#3b82f6">
        ~3800(论文实测,7.4×)
      </text>

      <g transform={`translate(${PAD_L + 10}, ${PAD_T + 10})`}>
        <line x1={0} y1={0} x2={20} y2={0} stroke="#9ca3af" strokeWidth={1.6} strokeDasharray="5 3" />
        <text x={26} y={4} fontSize={9} fill="#374151">原版 Transformer(固定窗口)</text>
        <line x1={0} y1={16} x2={20} y2={16} stroke="#3b82f6" strokeWidth={2} />
        <text x={26} y={20} fontSize={9} fill="#374151">Transformer-XL(段级循环累积)</text>
      </g>
    </svg>
  );
}
