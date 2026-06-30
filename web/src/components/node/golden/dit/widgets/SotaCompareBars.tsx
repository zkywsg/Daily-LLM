import { SOTA_COMPARE } from "../lib/data";

const W = 700;
const H = 240;

export function SotaCompareBars() {
  const PAD_L = 130;
  const PAD_R = 40;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 38;

  const yMax = 4.5;
  const wOf = (f: number) => (1 - f / yMax) * plotW; // 越低越长(因为 FID 越低越好,我们反向画)
  // actually let's just bar of fid scaled
  const barOf = (f: number) => (f / yMax) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DiT vs U-Net SOTA bars">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        ImageNet 256 class-conditional SOTA — DiT vs U-Net
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        FID-50K(越低越好)· 同等 FLOPs 下 DiT 比 U-Net 路线低 30-40%
      </text>

      {SOTA_COMPARE.map((r, i) => {
        const y = PAD_T + i * (rowH + 4);
        const w = barOf(r.fid);
        const color = r.isDit ? "#ec4899" : "#9ca3af";
        const bg = r.isDit ? "#fce7f3" : "#f3f4f6";
        return (
          <g key={i}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="#374151">{r.model}</text>
            <text x={PAD_L - 8} y={y + rowH / 2 + 18} textAnchor="end" fontSize={9} fill="#9ca3af">{r.params}M · {r.gflops} Gflops</text>

            <rect x={PAD_L} y={y} width={w} height={rowH - 4} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + w + 6} y={y + rowH / 2 + 2} fontSize={11} fontWeight={700} fill={color}>FID {r.fid.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}
