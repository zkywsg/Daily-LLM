import { TRADEOFF_TABLE } from "../lib/data";

const W = 700;
const H = 400;

export function ParamPerformanceScatterChart() {
  const PAD_L = 70;
  const PAD_B = 50;
  const PAD_T = 50;
  const plotW = 520;
  const plotH = 260;

  const maxParams = 120;
  const minRetain = 90;
  const maxRetain = 101;

  const xOf = (params: number) => PAD_L + (params / maxParams) * plotW;
  const yOf = (retain: number) => PAD_T + plotH - ((retain - minRetain) / (maxRetain - minRetain)) * plotH;
  const rOf = (speedup: number) => 8 + speedup * 8;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="参数量 vs GLUE 性能保留率 vs 推理速度散点图,DistilBERT vs BERT vs ALBERT">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        参数量 × 性能保留 × 推理速度 — 气泡大小 = 推理速度倍数
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <text x={PAD_L + plotW / 2} y={PAD_T + plotH + 35} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        参数量(M)
      </text>
      <text x={20} y={PAD_T + plotH / 2} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`}>
        GLUE 性能保留 %
      </text>

      {[90, 95, 100].map((v) => (
        <g key={v}>
          <line x1={PAD_L} y1={yOf(v)} x2={PAD_L + plotW} y2={yOf(v)} stroke="#f3f4f6" strokeWidth={1} />
          <text x={PAD_L - 8} y={yOf(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">{v}</text>
        </g>
      ))}

      {TRADEOFF_TABLE.map((row) => {
        const cx = xOf(row.paramsM);
        const cy = yOf(row.glueRetainPct);
        const isDistil = row.model === "DistilBERT";
        const color = isDistil ? "#10b981" : row.model === "ALBERT-base" ? "#f59e0b" : "#3b82f6";
        const bg = isDistil ? "#ecfdf5" : row.model === "ALBERT-base" ? "#fef3c7" : "#dbeafe";
        return (
          <g key={row.model}>
            <circle cx={cx} cy={cy} r={rOf(row.speedupX)} fill={bg} stroke={color} strokeWidth={1.8} fillOpacity={0.85} />
            <text x={cx} y={cy - rOf(row.speedupX) - 6} textAnchor="middle" fontSize={10} fontWeight={700} fill={color}>
              {row.model}
            </text>
            <text x={cx} y={cy + 3} textAnchor="middle" fontSize={8} fill={color}>
              {row.speedupX}x
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        DistilBERT 是唯一同时改善参数、速度、性能三个维度的方案;ALBERT 省参数但 forward 仍走满层,速度不变
      </text>
    </svg>
  );
}
