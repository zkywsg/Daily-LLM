import { SOTA_COMPARE } from "../lib/data";

const W = 700;
const H = 280;

export function SotaCompareBars() {
  const PAD_L = 150;
  const PAD_R = 60;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 36;

  const maxFid = 14;
  const wOf = (f: number) => (f / maxFid) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="2022 text-to-image SOTA comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        2022 年 COCO zero-shot FID-30K 对比(越低越好)
      </text>

      {SOTA_COMPARE.map((r, i) => {
        const y = PAD_T + i * (rowH + 6);
        const color = r.isImagen ? "#10b981" : "#9ca3af";
        const bg = r.isImagen ? "#ecfdf5" : "#f3f4f6";
        return (
          <g key={r.model}>
            <text x={PAD_L - 8} y={y + rowH / 2 - 2} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{r.model}</text>
            <text x={PAD_L - 8} y={y + rowH / 2 + 12} textAnchor="end" fontSize={9} fill="#9ca3af">{r.org}</text>
            <rect x={PAD_L} y={y} width={wOf(r.fid)} height={rowH - 6} fill={bg} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(r.fid) + 6} y={y + rowH / 2 + 2} fontSize={11} fontWeight={700} fill={color}>{r.fid.toFixed(2)}</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        Imagen 7.27 大幅领先 · 39.2% 人工偏好胜过真实照片(50% 是不可区分)
      </text>
    </svg>
  );
}
