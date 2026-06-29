import { discardProb, frequency, WORD_FREQS } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  t: number;  // threshold, default 1e-5
}

// 横轴 log(freq),纵轴 P_discard = 1 - sqrt(t/f)
export function SubsamplingCurve({ t }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // x 范围: log10(1e-7) ~ log10(1e-1)
  const logMin = -7;
  const logMax = -1;
  const xOf = (f: number) => PAD_L + ((Math.log10(f) - logMin) / (logMax - logMin)) * plotW;
  const yOf = (p: number) => PAD_T + (1 - p) * plotH;

  // curve points
  const pts: string[] = [];
  for (let i = 0; i <= 200; i++) {
    const lf = logMin + (logMax - logMin) * (i / 200);
    const f = Math.pow(10, lf);
    const p = discardProb(f, t);
    pts.push(`${xOf(f)},${yOf(p)}`);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Subsampling discard probability curve">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        高频词丢弃概率 P_discard = 1 − √(t / f), t = {t.toExponential(0)}
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* y ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((p) => (
        <g key={p}>
          <line x1={PAD_L - 4} y1={yOf(p)} x2={PAD_L} y2={yOf(p)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(p) + 4} textAnchor="end" fontSize={10} fill="#6b7280">{p.toFixed(2)}</text>
          <line x1={PAD_L} y1={yOf(p)} x2={W - PAD_R} y2={yOf(p)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}
      {/* x ticks */}
      {[-7, -6, -5, -4, -3, -2, -1].map((lf) => (
        <g key={lf}>
          <line x1={xOf(Math.pow(10, lf))} y1={PAD_T + plotH} x2={xOf(Math.pow(10, lf))} y2={PAD_T + plotH + 4} stroke="#9ca3af" />
          <text x={xOf(Math.pow(10, lf))} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={9} fill="#6b7280">10^{lf}</text>
        </g>
      ))}

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={10} fill="#6b7280">word frequency f(w) (log scale)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">P_discard</text>

      {/* threshold line */}
      <line x1={xOf(t)} y1={PAD_T} x2={xOf(t)} y2={PAD_T + plotH} stroke="#3b82f6" strokeDasharray="4 4" strokeWidth={1.2} />
      <text x={xOf(t) + 6} y={PAD_T + 14} fontSize={10} fill="#1e40af" fontWeight={600}>t = {t.toExponential(0)}</text>

      {/* curve */}
      <polyline points={pts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.2} />

      {/* word markers */}
      {WORD_FREQS.map((w) => {
        const f = frequency(w.word);
        if (f < Math.pow(10, logMin)) return null;
        const p = discardProb(f, t);
        const cx = xOf(f);
        const cy = yOf(p);
        const color = w.category === "stopword" ? "#ec4899" : w.category === "rare" ? "#3b82f6" : "#f59e0b";
        return (
          <g key={w.word}>
            <circle cx={cx} cy={cy} r={4} fill={color} stroke="#fff" strokeWidth={1} />
            <text x={cx + 6} y={cy - 6} fontSize={9} fontWeight={600} fill={color}>{w.word}</text>
          </g>
        );
      })}
    </svg>
  );
}
