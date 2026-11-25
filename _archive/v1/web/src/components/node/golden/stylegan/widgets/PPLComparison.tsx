import { PPL_COMPARE } from "../lib/data";

const W = 700;
const H = 280;

// Z vs W PPL 条形对比 (越低越好 = 越"线性")
export function PPLComparison() {
  const PAD_L = 110;
  const PAD_R = 40;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 34;
  const gap = 6;

  const maxV = 450;
  const wOf = (v: number) => (v / maxV) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PPL comparison Z vs W">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Perceptual Path Length — W 空间线性度是 Z 的两倍
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        PPL 越低 → latent 空间越"光滑" → 越适合做 editing
      </text>

      {PPL_COMPARE.map((row, i) => {
        const y = PAD_T + i * (rowH * 2 + gap * 2);
        return (
          <g key={row.space}>
            <text x={PAD_L - 8} y={y + rowH + 4} textAnchor="end" fontSize={11} fontWeight={700} fill={row.color}>{row.space}</text>

            {/* full */}
            <rect x={PAD_L} y={y} width={wOf(row.full)} height={rowH - 4}
                  fill={row.color} fillOpacity={0.15} stroke={row.color} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + 8} y={y + (rowH - 4) / 2 + 4} fontSize={10} fill="#374151">full</text>
            <text x={PAD_L + wOf(row.full) + 6} y={y + (rowH - 4) / 2 + 4} fontSize={10} fontWeight={700} fill={row.color}>{row.full.toFixed(1)}</text>

            {/* end */}
            <rect x={PAD_L} y={y + rowH} width={wOf(row.end)} height={rowH - 4}
                  fill={row.color} fillOpacity={0.15} stroke={row.color} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + 8} y={y + rowH + (rowH - 4) / 2 + 4} fontSize={10} fill="#374151">end</text>
            <text x={PAD_L + wOf(row.end) + 6} y={y + rowH + (rowH - 4) / 2 + 4} fontSize={10} fontWeight={700} fill={row.color}>{row.end.toFixed(1)}</text>
          </g>
        );
      })}
    </svg>
  );
}
