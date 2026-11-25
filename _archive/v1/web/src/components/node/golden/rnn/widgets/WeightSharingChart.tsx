import { sharedParams, unsharedParams } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  seqLen: number;
}

export function WeightSharingChart({ seqLen }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const hiddenDim = 128, inputDim = 128;
  const shared = sharedParams(hiddenDim, inputDim);

  const maxLen = 200;
  const xOf = (l: number) => PAD_L + (l / maxLen) * plotW;
  const maxParams = unsharedParams(hiddenDim, inputDim, maxLen);
  const yOf = (p: number) => PAD_T + (1 - p / maxParams) * plotH;

  const sharedPts: string[] = [];
  const unsharedPts: string[] = [];
  for (let i = 0; i <= 100; i++) {
    const l = (i / 100) * maxLen;
    sharedPts.push(`${xOf(l)},${yOf(shared)}`);
    unsharedPts.push(`${xOf(l)},${yOf(unsharedParams(hiddenDim, inputDim, l))}`);
  }

  const curUnshared = unsharedParams(hiddenDim, inputDim, seqLen);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Weight sharing parameter count comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        权重共享 — 参数量与序列长度无关
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, seqLen, maxLen].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b).map((l) => (
        <g key={l}>
          <line x1={xOf(l)} y1={PAD_T + plotH} x2={xOf(l)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(l)} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={9} fill="#374151">{l}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">序列长度 T</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">参数量</text>

      <polyline points={sharedPts.join(" ")} fill="none" stroke="#10b981" strokeWidth={2.4} />
      <polyline points={unsharedPts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.4} />

      <line x1={xOf(seqLen)} y1={PAD_T} x2={xOf(seqLen)} y2={PAD_T + plotH} stroke="#1f2937" strokeWidth={1.2} strokeDasharray="3 3" />
      <circle cx={xOf(seqLen)} cy={yOf(shared)} r={5} fill="#10b981" stroke="#fff" strokeWidth={1.5} />
      <circle cx={xOf(seqLen)} cy={yOf(curUnshared)} r={5} fill="#ec4899" stroke="#fff" strokeWidth={1.5} />

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#10b981" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={10} fontWeight={600} fill="#10b981">共享权重(RNN)</text>
        <line x1={150} y1={6} x2={170} y2={6} stroke="#ec4899" strokeWidth={2.5} />
        <text x={176} y={10} fontSize={10} fontWeight={600} fill="#ec4899">不共享(假想 per-step W)</text>
      </g>

      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        T={seqLen}: 共享 {shared.toLocaleString()} 参数 vs 不共享 {curUnshared.toLocaleString()} 参数(差 {seqLen}×)
      </text>
    </svg>
  );
}
