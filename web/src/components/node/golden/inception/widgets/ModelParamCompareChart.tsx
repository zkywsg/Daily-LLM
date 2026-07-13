import { MODEL_PARAM_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function ModelParamCompareChart() {
  const PAD_L = 130;
  const PAD_T = 44;
  const barMaxW = 470;
  const rowH = 48;
  const maxParams = Math.max(...MODEL_PARAM_COMPARE.map((r) => r.params));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="AlexNet / ZFNet / VGG-16 / GoogLeNet 整网参数量对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        整网参数量对比 — GoogLeNet 仅 5M,VGG-16 的 1/28
      </text>

      {MODEL_PARAM_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = (row.params / maxParams) * barMaxW;
        const fill = row.highlight ? "#10b981" : "#9ca3af";
        const bg = row.highlight ? "#ecfdf5" : "#f3f4f6";
        return (
          <g key={`${row.label}-${i}`}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={11} fill="var(--ink-secondary)">
              {row.label}
            </text>
            <text x={PAD_L - 10} y={y + 30} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {row.year} · Top-5 {row.top5Error}%
            </text>
            <rect x={PAD_L} y={y} width={barMaxW} height={26} fill={bg} rx={3} />
            <rect x={PAD_L} y={y} width={Math.max(w, 4)} height={26} fill={fill} rx={3} />
            <text x={PAD_L + Math.max(w, 4) + 8} y={y + 18} fontSize={12} fontWeight={700} fill="var(--ink-primary)">
              {(row.params / 1_000_000).toLocaleString("en-US")}M
            </text>
          </g>
        );
      })}
    </svg>
  );
}
