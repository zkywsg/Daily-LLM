import { ENCODER_SCALING } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightIdx: number;
}

export function EncoderScalingChart({ highlightIdx }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 70;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const logMin = Math.log10(0.05);
  const logMax = Math.log10(15);
  const xOf = (p: number) => PAD_L + ((Math.log10(p) - logMin) / (logMax - logMin)) * plotW;

  const yMin = 6, yMax = 14;
  const yOf = (f: number) => PAD_T + ((f - yMin) / (yMax - yMin)) * plotH;

  const pts = ENCODER_SCALING.map((e) => `${xOf(e.params)},${yOf(e.fid)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Text encoder scaling vs FID">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        文本编码器规模 vs COCO FID — 越大越好,但不是 diffusion 模型
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[7, 9, 11, 13].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}
      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={10} fill="#6b7280">text encoder 参数(B, log scale)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">COCO FID-30K(越低越好)</text>

      <polyline points={pts} fill="none" stroke="#ec4899" strokeWidth={2} strokeDasharray="4 3" opacity={0.5} />

      {ENCODER_SCALING.map((e, i) => {
        const isHigh = i === highlightIdx || highlightIdx === -1;
        const isCLIP = e.name.includes("CLIP");
        const color = isCLIP ? "#9ca3af" : e.name === "T5-XXL" ? "#10b981" : "#ec4899";
        return (
          <g key={e.name} opacity={isHigh ? 1 : 0.35}>
            <circle cx={xOf(e.params)} cy={yOf(e.fid)} r={isHigh ? 7 : 5} fill={color} stroke="#fff" strokeWidth={1.5} />
            <text x={xOf(e.params)} y={yOf(e.fid) - 12} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>{e.name}</text>
            <text x={xOf(e.params)} y={yOf(e.fid) + 20} textAnchor="middle" fontSize={9} fill="#374151">{e.fid.toFixed(1)}</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        T5-XXL 比 CLIP 大 28× · FID 从 12.1 降到 7.27,改善 40%
      </text>
    </svg>
  );
}
