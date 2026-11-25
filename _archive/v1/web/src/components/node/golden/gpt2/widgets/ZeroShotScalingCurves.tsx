import { ZS_SCALING, SCALE_BREAKPOINTS } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  selectedTask: number;
}

// 4 个任务在 4 个 GPT-2 规模上的 zero-shot 表现
export function ZeroShotScalingCurves({ selectedTask }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // log scale x (params)
  const logMin = Math.log10(100);
  const logMax = Math.log10(1700);
  const xOf = (p: number) => PAD_L + ((Math.log10(p) - logMin) / (logMax - logMin)) * plotW;

  // 归一化 y:对每个任务把 vals 拉伸到 [0, 1] 用于绘图
  function normalize(vals: number[], reverse: boolean): number[] {
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const span = mx - mn || 1;
    return vals.map((v) => reverse ? (mx - v) / span : (v - mn) / span);
  }
  const yOf = (norm: number) => PAD_T + (1 - norm) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Zero-shot performance vs model scale">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Zero-shot 性能 vs 模型规模 — 4 个 GPT-2 size,4 个任务
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* x ticks at the 4 model sizes */}
      {SCALE_BREAKPOINTS.map((p, i) => (
        <g key={p}>
          <line x1={xOf(p)} y1={PAD_T + plotH} x2={xOf(p)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(p)} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={10} fontWeight={500} fill="#374151">
            {p < 1000 ? `${p}M` : `${(p / 1000).toFixed(1)}B`}
          </text>
          <text x={xOf(p)} y={PAD_T + plotH + 30} textAnchor="middle" fontSize={9} fill="#9ca3af">
            {["Small", "Medium", "Large", "XL"][i]}
          </text>
        </g>
      ))}

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="#6b7280">model parameters (log)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">归一化 zero-shot 性能</text>

      {ZS_SCALING.map((t, i) => {
        const reverse = t.task.includes("↓");
        const normVals = normalize(t.vals, reverse);
        const pts = normVals.map((v, k) => `${xOf(SCALE_BREAKPOINTS[k])},${yOf(v)}`).join(" ");
        const isSelected = i === selectedTask || selectedTask === -1;
        return (
          <g key={i} opacity={isSelected ? 1 : 0.25}>
            <polyline points={pts} fill="none" stroke={t.color} strokeWidth={isSelected ? 2.6 : 1.6} />
            {normVals.map((v, k) => (
              <circle key={k} cx={xOf(SCALE_BREAKPOINTS[k])} cy={yOf(v)} r={isSelected ? 5 : 3.5} fill={t.color} stroke="#fff" strokeWidth={1.5} />
            ))}
            {isSelected && normVals.map((v, k) => (
              <text key={k} x={xOf(SCALE_BREAKPOINTS[k]) + 8} y={yOf(v) - 8} fontSize={9} fontWeight={600} fill={t.color}>
                {t.vals[k].toFixed(1)}
              </text>
            ))}
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 10})`}>
        {ZS_SCALING.map((t, i) => (
          <g key={i} transform={`translate(0, ${i * 18})`} opacity={i === selectedTask || selectedTask === -1 ? 1 : 0.4}>
            <line x1={0} y1={6} x2={20} y2={6} stroke={t.color} strokeWidth={2.5} />
            <text x={26} y={10} fontSize={10} fontWeight={600} fill={t.color}>
              {t.task} {t.isEmergent && "(涌现型)"}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}
