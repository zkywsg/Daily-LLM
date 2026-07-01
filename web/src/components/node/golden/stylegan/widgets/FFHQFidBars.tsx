import { FFHQ_FID, LSUN } from "../lib/data";

const W = 700;
const H = 320;

// 上半:FFHQ 三条 · 下半:LSUN 三类 progressive vs stylegan
export function FFHQFidBars() {
  const PAD_L = 130;
  const PAD_R = 40;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;

  const maxFid = 40;
  const barOf = (f: number) => (f / maxFid) * plotW;

  const rowH = 20;
  const startY = PAD_T;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="StyleGAN FID benchmarks">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        FFHQ + LSUN FID(越低越好)
      </text>

      {/* FFHQ */}
      <text x={PAD_L - 8} y={startY - 4} textAnchor="end" fontSize={10} fontWeight={700} fill="#374151" style={{ textTransform: "uppercase" }}>FFHQ 人脸</text>
      {FFHQ_FID.map((r, i) => {
        const y = startY + i * (rowH + 4);
        const color = r.isStyleGan ? "#ec4899" : "#9ca3af";
        const bg = r.isStyleGan ? "#fce7f3" : "#f3f4f6";
        return (
          <g key={r.model}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={10} fontWeight={600} fill="#374151">{r.model}</text>
            <rect x={PAD_L} y={y} width={barOf(r.fid)} height={rowH - 3}
                  fill={bg} stroke={color} strokeWidth={1.2} rx={2} />
            <text x={PAD_L + barOf(r.fid) + 6} y={y + rowH / 2 + 4} fontSize={10} fontWeight={700} fill={color}>{r.fid.toFixed(2)}</text>
          </g>
        );
      })}

      {/* LSUN */}
      <text x={PAD_L - 8} y={startY + FFHQ_FID.length * (rowH + 4) + 20} textAnchor="end"
            fontSize={10} fontWeight={700} fill="#374151" style={{ textTransform: "uppercase" }}>LSUN 场景</text>
      {LSUN.map((r, i) => {
        const y = startY + FFHQ_FID.length * (rowH + 4) + 28 + i * (rowH * 2 + 4);
        return (
          <g key={r.cls}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={10} fontWeight={600} fill="#374151">{r.cls} Prog</text>
            <rect x={PAD_L} y={y} width={barOf(r.prog)} height={rowH - 3}
                  fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + barOf(r.prog) + 6} y={y + rowH / 2 + 4} fontSize={10} fill="#9ca3af">{r.prog.toFixed(2)}</text>

            <text x={PAD_L - 8} y={y + rowH + rowH / 2 + 4} textAnchor="end" fontSize={10} fontWeight={600} fill="#ec4899">{r.cls} Style</text>
            <rect x={PAD_L} y={y + rowH} width={barOf(r.stylegan)} height={rowH - 3}
                  fill="#fce7f3" stroke="#ec4899" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + barOf(r.stylegan) + 6} y={y + rowH + rowH / 2 + 4} fontSize={10} fontWeight={700} fill="#ec4899">{r.stylegan.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}
