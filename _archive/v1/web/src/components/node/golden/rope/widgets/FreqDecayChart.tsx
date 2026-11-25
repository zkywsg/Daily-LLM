import { decayAtDistance, buildFreqDims } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightFreq: number | null;
}

export function FreqDecayChart({ highlightFreq }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const maxDist = 512;
  const xOf = (d: number) => PAD_L + (d / maxDist) * plotW;
  const yOf = (v: number) => PAD_T + (1 - (v + 1) / 2) * plotH;

  const pts: string[] = [];
  for (let i = 0; i <= 200; i++) {
    const dist = (i / 200) * maxDist;
    pts.push(`${xOf(dist)},${yOf(decayAtDistance(dist))}`);
  }

  const freqDims = buildFreqDims();

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RoPE frequency decay curve">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        长程衰减 — 多频率求和后内积随距离振荡衰减
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={yOf(0)} x2={W - PAD_R} y2={yOf(0)} stroke="#e5e7eb" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[-1, 0, 1].map((v) => (
        <g key={v}>
          <line x1={PAD_L - 4} y1={yOf(v)} x2={PAD_L} y2={yOf(v)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(v) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{v}</text>
        </g>
      ))}
      {[0, 128, 256, 384, 512].map((d) => (
        <g key={d}>
          <line x1={xOf(d)} y1={PAD_T + plotH} x2={xOf(d)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(d)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{d}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={10} fill="#6b7280">相对距离 |m-n|</text>

      <polyline points={pts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.4} />

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        高频维度快速振荡抵消(短距专用)· 低频维度慢慢衰减(长距专用)· 不需要显式 attention mask
      </text>

      {/* freq dims 侧栏放在右上角小表 */}
      <g transform={`translate(${W - 190}, ${PAD_T + 6})`}>
        {freqDims.map((f, i) => (
          <g key={i} transform={`translate(0, ${i * 16})`} opacity={highlightFreq === f.index ? 1 : 0.6}>
            <text x={0} y={0} fontSize={9} fontWeight={highlightFreq === f.index ? 700 : 500} fill="#374151">
              {f.name}: θ={f.theta.toExponential(1)} ({f.role})
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}
