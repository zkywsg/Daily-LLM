import { BENCH_ROWS } from "../lib/data";

const W = 700;
const ROW_H = 34;
const H = 60 + BENCH_ROWS.length * ROW_H;

// 论文 Table 1:参数量相近但 top-1 差异巨大,DeiT 系列训练成本最低。
export function BenchmarkBars() {
  const PAD_L = 190;
  const PAD_R = 70;
  const plotW = W - PAD_L - PAD_R;
  const maxV = 85;
  const minV = 74;
  const wOf = (v: number) => ((v - minV) / (maxV - minV)) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ImageNet top-1 benchmark comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        ImageNet top-1 — DeiT 用最少算力打平 / 击败更贵的方案
      </text>

      {BENCH_ROWS.map((r, i) => {
        const y = 40 + i * ROW_H;
        const fill = r.highlight ? "#ec4899" : "#e5e7eb";
        const stroke = r.highlight ? "#db2777" : "#9ca3af";
        const textColor = r.highlight ? "#9d174d" : "#4b5563";
        return (
          <g key={r.model}>
            <text x={PAD_L - 8} y={y + 16} textAnchor="end" fontSize={10} fontWeight={r.highlight ? 700 : 500} fill={textColor}>
              {r.model}
            </text>
            <rect x={PAD_L} y={y} width={wOf(r.top1)} height={22} rx={3} fill={fill} stroke={stroke} strokeWidth={1.2} />
            <text x={PAD_L + wOf(r.top1) + 6} y={y + 16} fontSize={10} fontWeight={700} fill={textColor}>
              {r.top1}
            </text>
            <text x={W - PAD_R + 8} y={y + 16} fontSize={9} fill="var(--ink-muted)">
              {r.hardware} · {r.trainTime}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
