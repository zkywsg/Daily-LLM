import { ACCURACY_VS_N } from "../lib/data";

const W = 700;
const H = 320;

export function AccuracyVsSamplesChart() {
  const PAD_L = 60;
  const PAD_T = 40;
  const plotW = 580;
  const plotH = 200;
  const minAcc = 50;
  const maxAcc = 80;
  const maxN = 40;

  // log-ish scale on x (N=1..40), evenly spaced by data point since only 5 known samples
  const xOf = (i: number) => PAD_L + (i / (ACCURACY_VS_N.length - 1)) * plotW;
  const yOf = (acc: number) => PAD_T + plotH - ((acc - minAcc) / (maxAcc - minAcc)) * plotH;

  const points = ACCURACY_VS_N.map((row, i) => `${xOf(i)},${yOf(row.gsm8k)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="采样次数 N 越大,GSM8K 准确率越高,但呈对数线性增长,收益递减"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Test-time compute scaling:N ↑ → 准确率 ↑(对数线性,收益递减)
      </text>

      {[50, 60, 70, 80].map((tick) => (
        <g key={tick}>
          <line x1={PAD_L} y1={yOf(tick)} x2={PAD_L + plotW} y2={yOf(tick)} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2,3" />
          <text x={PAD_L - 10} y={yOf(tick) + 4} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
            {tick}%
          </text>
        </g>
      ))}

      <polyline points={points} fill="none" stroke="#ec4899" strokeWidth={2.5} />

      {ACCURACY_VS_N.map((row, i) => (
        <g key={row.n}>
          <circle cx={xOf(i)} cy={yOf(row.gsm8k)} r={5} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
          <text x={xOf(i)} y={yOf(row.gsm8k) - 14} textAnchor="middle" fontSize={10} fontWeight={700} fill="#9d174d">
            {row.gsm8k}%
          </text>
          <text x={xOf(i)} y={PAD_T + plotH + 20} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
            N={row.n}
          </text>
          <text x={xOf(i)} y={PAD_T + plotH + 34} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
            {row.cost}算力
          </text>
        </g>
      ))}

      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        GSM8K,PaLM 540B。40 次采样比 1 次贵 40×,准确率仅 +17.9 分——精度/算力 trade-off 递减
      </text>
    </svg>
  );
}
