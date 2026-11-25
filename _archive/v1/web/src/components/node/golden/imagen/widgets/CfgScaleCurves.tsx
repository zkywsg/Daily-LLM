import { cfgFidelity, cfgCreativity, cfgDistortion } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  w: number;
}

export function CfgScaleCurves({ w }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const wMin = 0, wMax = 20;
  const xOf = (ww: number) => PAD_L + ((ww - wMin) / (wMax - wMin)) * plotW;
  const yOf = (v: number) => PAD_T + (1 - v / 100) * plotH;

  const fidPts: string[] = [];
  const crePts: string[] = [];
  const disPts: string[] = [];
  for (let i = 0; i <= 100; i++) {
    const ww = wMin + (i / 100) * (wMax - wMin);
    fidPts.push(`${xOf(ww)},${yOf(cfgFidelity(ww))}`);
    crePts.push(`${xOf(ww)},${yOf(cfgCreativity(ww))}`);
    disPts.push(`${xOf(ww)},${yOf(cfgDistortion(ww))}`);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CFG scale trade-off curves">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        cfg_scale trade-off — w = {w.toFixed(1)}
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 25, 50, 75, 100].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}%</text>
        </g>
      ))}
      {[0, 5, 10, 15, 20].map((ww) => (
        <g key={ww}>
          <line x1={xOf(ww)} y1={PAD_T + plotH} x2={xOf(ww)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(ww)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{ww}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">w (cfg_scale)</text>

      <polyline points={fidPts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.4} />
      <polyline points={crePts.join(" ")} fill="none" stroke="#3b82f6" strokeWidth={2.4} />
      <polyline points={disPts.join(" ")} fill="none" stroke="#f59e0b" strokeWidth={2.4} />

      {/* current w marker */}
      <line x1={xOf(w)} y1={PAD_T} x2={xOf(w)} y2={PAD_T + plotH} stroke="#1f2937" strokeWidth={1.5} strokeDasharray="4 3" />
      <circle cx={xOf(w)} cy={yOf(cfgFidelity(w))} r={5} fill="#ec4899" stroke="#fff" strokeWidth={1.5} />
      <circle cx={xOf(w)} cy={yOf(cfgCreativity(w))} r={5} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
      <circle cx={xOf(w)} cy={yOf(cfgDistortion(w))} r={5} fill="#f59e0b" stroke="#fff" strokeWidth={1.5} />

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#ec4899" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={10} fontWeight={600} fill="#ec4899">prompt fidelity</text>
        <line x1={150} y1={6} x2={170} y2={6} stroke="#3b82f6" strokeWidth={2.5} />
        <text x={176} y={10} fontSize={10} fontWeight={600} fill="#3b82f6">creativity</text>
        <line x1={280} y1={6} x2={300} y2={6} stroke="#f59e0b" strokeWidth={2.5} />
        <text x={306} y={10} fontSize={10} fontWeight={600} fill="#f59e0b">distortion</text>
      </g>

      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {w < 3 ? "w 太低:图与 prompt 无关" : w <= 10 ? "w=7-10:典型默认,fidelity + quality 平衡" : "w>15:过度放大,over-saturation 失真"}
      </text>
    </svg>
  );
}
