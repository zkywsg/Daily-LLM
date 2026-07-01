import { BLEU_VS_LENGTH } from "../lib/data";

const W = 700;
const H = 300;

export function BleuVsLengthCurve() {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xMin = 10, xMax = 70;
  const yMin = 20, yMax = 38;
  const xOf = (l: number) => PAD_L + ((l - xMin) / (xMax - xMin)) * plotW;
  const yOf = (b: number) => PAD_T + ((yMax - b) / (yMax - yMin)) * plotH;

  const pts = BLEU_VS_LENGTH.map((p) => `${xOf(p.length)},${yOf(p.bleu)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BLEU vs source sentence length">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        信息瓶颈实证 — BLEU 随源句长度急剧下降
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[20, 25, 30, 35].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}
      {[10, 30, 50, 70].map((l) => (
        <g key={l}>
          <line x1={xOf(l)} y1={PAD_T + plotH} x2={xOf(l)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(l)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{l}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="#6b7280">源句长度(词)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">BLEU</text>

      <polyline points={pts} fill="none" stroke="#ec4899" strokeWidth={2.5} />
      {BLEU_VS_LENGTH.map((p, i) => (
        <circle key={i} cx={xOf(p.length)} cy={yOf(p.bleu)} r={5} fill="#ec4899" stroke="#fff" strokeWidth={1.5} />
      ))}

      <text x={W / 2} y={H - 24} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        &lt;20 词 BLEU≈35,70 词跌到 24 — 一个 500 维向量装不下长句所有语义
      </text>
    </svg>
  );
}
