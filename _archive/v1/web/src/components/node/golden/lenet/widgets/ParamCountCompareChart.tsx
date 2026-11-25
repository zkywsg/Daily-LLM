import { PARAM_COMPARE } from "../lib/data";

const W = 700;
const H = 180;

export function ParamCountCompareChart() {
  const PAD_L = 260;
  const PAD_T = 40;
  const barMaxW = 340;
  const rowH = 44;
  const maxParams = Math.max(...PARAM_COMPARE.map((r) => r.params));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="LeNet-5 整网参数量与同输入接 MLP 第一层参数量对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        参数量对比 — LeNet-5 整网 vs MLP 仅第一层
      </text>

      {PARAM_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = (row.params / maxParams) * barMaxW;
        const fill = row.highlight ? "#10b981" : "#9ca3af";
        const bg = row.highlight ? "#ecfdf5" : "#f3f4f6";
        return (
          <g key={row.label}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={10} fill="var(--ink-secondary)">
              {row.label}
            </text>
            <rect x={PAD_L} y={y} width={barMaxW} height={22} fill={bg} rx={3} />
            <rect x={PAD_L} y={y} width={Math.max(w, 4)} height={22} fill={fill} rx={3} />
            <text x={PAD_L + Math.max(w, 4) + 8} y={y + 16} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.params.toLocaleString("en-US")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
