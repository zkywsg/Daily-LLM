import { METHOD_COMPARE } from "../lib/data";

const W = 700;
const H = 320;

// 4 个维度:模型数 / hparams / 代码行数(log) / 训练成本
export function MethodComplexityBars() {
  const dims: Array<{ name: string; max: number; field: keyof typeof METHOD_COMPARE[0]; format: (v: number) => string; useLog?: boolean }> = [
    { name: "模型数",   max: 4,    field: "models",    format: (v) => `${v} 个` },
    { name: "Hparam 数", max: 14, field: "hparams",   format: (v) => `${v} 个` },
    { name: "代码行数", max: 5000, field: "codeLines", format: (v) => v >= 1000 ? `${(v / 1000).toFixed(1)}K` : `${v}`, useLog: true },
    { name: "相对成本", max: 1.0,  field: "cost",      format: (v) => v >= 0.1 ? `${(v * 100).toFixed(0)}%` : `${(v * 100).toFixed(0)}%` },
  ];

  const PAD_L = 90;
  const PAD_R = 40;
  const PAD_T = 50;
  const PAD_B = 30;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const rowH = plotH / dims.length - 8;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PPO vs DPO complexity bars">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        PPO vs DPO — 4 维度工程复杂度对比
      </text>

      {dims.map((d, di) => {
        const y = PAD_T + di * (rowH + 8);
        return (
          <g key={di}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="#374151">{d.name}</text>
            {METHOD_COMPARE.map((m, mi) => {
              const v = m[d.field] as number;
              const norm = d.useLog
                ? Math.log10(v + 1) / Math.log10(d.max + 1)
                : v / d.max;
              const w = norm * plotW;
              const yy = y + mi * (rowH / 2 + 1);
              const color = m.isDpo ? "#10b981" : "#ec4899";
              const bg = m.isDpo ? "#ecfdf5" : "#fce7f3";
              return (
                <g key={mi}>
                  <rect x={PAD_L} y={yy} width={Math.max(w, 4)} height={rowH / 2 - 2} fill={bg} stroke={color} strokeWidth={1.2} rx={2} />
                  <text x={PAD_L + Math.max(w, 4) + 6} y={yy + (rowH / 2 - 2) / 2 + 4} fontSize={10} fontWeight={700} fill={color}>
                    {d.format(v)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${PAD_L}, ${H - 18})`}>
        <rect x={0} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={18} y={10} fontSize={10} fill="#374151">PPO (InstructGPT)</text>
        <rect x={150} y={0} width={12} height={12} fill="#ecfdf5" stroke="#10b981" />
        <text x={168} y={10} fontSize={10} fill="#374151">DPO</text>
      </g>
    </svg>
  );
}
