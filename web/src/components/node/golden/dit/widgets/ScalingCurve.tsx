import { DIT_SIZES } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightIdx: number;
}

// log(Gflops) vs FID — 几乎线性
export function ScalingCurve({ highlightIdx }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // log10 X
  const logMin = Math.log10(4);
  const logMax = Math.log10(200);
  const xOf = (g: number) => PAD_L + ((Math.log10(g) - logMin) / (logMax - logMin)) * plotW;

  const yMin = 10;
  const yMax = 75;
  const yOf = (f: number) => PAD_T + ((f - yMin) / (yMax - yMin)) * plotH;

  const pts = DIT_SIZES.map((s) => `${xOf(s.gflops)},${yOf(s.fid)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DiT scaling curve">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        DiT scaling — log(Gflops) vs FID(400K iter, ImageNet 256, cfg=1.5)
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* y ticks */}
      {[10, 20, 40, 60].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {/* x ticks (log) */}
      {[4, 10, 30, 100, 200].map((g) => (
        <g key={g}>
          <line x1={xOf(g)} y1={PAD_T + plotH} x2={xOf(g)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(g)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{g}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 18} textAnchor="middle" fontSize={10} fill="#6b7280">Gflops (log scale)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">FID-50K (lower is better)</text>

      {/* curve */}
      <polyline points={pts} fill="none" stroke="#ec4899" strokeWidth={2.5} />

      {DIT_SIZES.map((s, i) => {
        const isHigh = i === highlightIdx || highlightIdx === -1;
        return (
          <g key={i} opacity={isHigh ? 1 : 0.4}>
            <circle cx={xOf(s.gflops)} cy={yOf(s.fid)} r={i === highlightIdx ? 8 : 6} fill="#ec4899" stroke="#fff" strokeWidth={2} />
            <text x={xOf(s.gflops)} y={yOf(s.fid) - 14} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">
              {s.name}
            </text>
            <text x={xOf(s.gflops)} y={yOf(s.fid) + 22} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">
              FID {s.fid.toFixed(2)}
            </text>
            <text x={xOf(s.gflops)} y={yOf(s.fid) + 36} textAnchor="middle" fontSize={9} fill="#6b7280">
              {s.paramsM}M · {s.gflops}G
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        FID 随 log(Gflops) 几乎线性下降 — diffusion 第一次有干净的可外推 scaling
      </text>
    </svg>
  );
}
