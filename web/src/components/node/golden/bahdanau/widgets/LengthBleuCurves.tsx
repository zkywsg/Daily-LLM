import { LENGTH_BLEU } from "../lib/data";

const W = 700;
const H = 300;

export function LengthBleuCurves() {
  const PAD_L = 60;
  const PAD_R = 40;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const n = LENGTH_BLEU.length;
  const yMax = 32;
  const xOf = (i: number) => PAD_L + (i / (n - 1)) * plotW;
  const yOf = (v: number) => PAD_T + ((yMax - v) / yMax) * plotH;

  const fixedPts = LENGTH_BLEU.map((b, i) => `${xOf(i)},${yOf(b.fixedC)}`).join(" ");
  const attnPts  = LENGTH_BLEU.map((b, i) => `${xOf(i)},${yOf(b.attention)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Length-binned BLEU comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        按源句长度切分的 BLEU(WMT'14 EN→FR)
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 10, 20, 30].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {LENGTH_BLEU.map((b, i) => (
        <text key={i} x={xOf(i)} y={PAD_T + plotH + 18} textAnchor="middle" fontSize={11} fontWeight={500} fill="#374151">{b.bucket}</text>
      ))}
      <text x={W / 2} y={H - 18} textAnchor="middle" fontSize={10} fill="#6b7280">源句词数</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">BLEU</text>

      {/* curves */}
      <polyline points={fixedPts} fill="none" stroke="#ec4899" strokeWidth={2.5} />
      <polyline points={attnPts} fill="none" stroke="#10b981" strokeWidth={2.5} />

      {LENGTH_BLEU.map((b, i) => (
        <g key={i}>
          <circle cx={xOf(i)} cy={yOf(b.fixedC)} r={5} fill="#ec4899" stroke="#fff" strokeWidth={1.5} />
          <text x={xOf(i) - 10} y={yOf(b.fixedC) + 4} textAnchor="end" fontSize={10} fontWeight={600} fill="#ec4899">{b.fixedC}</text>
          <circle cx={xOf(i)} cy={yOf(b.attention)} r={5} fill="#10b981" stroke="#fff" strokeWidth={1.5} />
          <text x={xOf(i) + 10} y={yOf(b.attention) - 6} fontSize={10} fontWeight={600} fill="#10b981">{b.attention}</text>
        </g>
      ))}

      {/* legend */}
      <g transform={`translate(${PAD_L + 20}, ${PAD_T + 10})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#ec4899" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={10} fontWeight={600} fill="#ec4899">Seq2Seq (固定 c)</text>
        <line x1={130} y1={6} x2={150} y2={6} stroke="#10b981" strokeWidth={2.5} />
        <text x={156} y={10} fontSize={10} fontWeight={600} fill="#10b981">+ Attention</text>
      </g>
    </svg>
  );
}
