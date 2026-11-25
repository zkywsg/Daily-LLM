import { EMERGENT_TASKS, MODEL_LINEUP } from "../lib/scaling";

const W = 700;
const H = 300;
const PAD = { left: 60, right: 24, top: 36, bottom: 50 };

// Emergent ability:某些任务在参数过临界后突然跳变。
// 多条曲线叠加显示"涌现"现象——小模型几乎全 0,某个阈值后陡升。

const COLORS = ["#ec4899", "#3b82f6", "#f59e0b"];

export function EmergentAbilityCurve() {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const minLogN = 8;
  const maxLogN = 13;
  const xScale = (logN: number) => PAD.left + ((logN - minLogN) / (maxLogN - minLogN)) * innerW;
  const yScale = (acc: number) => PAD.top + (1 - acc) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Emergent abilities of large language models">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Emergent Abilities — 任务能力 vs 参数量(Wei 2022)
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* x 标 */}
      {[8, 9, 10, 11, 12, 13].map((logN) => (
        <g key={`x-${logN}`}>
          <line x1={xScale(logN)} y1={H - PAD.bottom} x2={xScale(logN)} y2={H - PAD.bottom + 4} stroke="var(--ink-muted)" />
          <text x={xScale(logN)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
            10^{logN}
          </text>
        </g>
      ))}
      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        参数量 N →
      </text>

      {/* y 标 */}
      {[0, 0.25, 0.5, 0.75, 1].map((a) => (
        <g key={`y-${a}`}>
          <line x1={PAD.left - 4} x2={W - PAD.right} y1={yScale(a)} y2={yScale(a)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 8} y={yScale(a) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {(a * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {/* 标 GPT-2/GPT-3 位置 */}
      {MODEL_LINEUP.filter((m) => m.name === "GPT-2" || m.name === "GPT-3").map((m) => {
        const logN = Math.log10(m.params);
        return (
          <g key={m.name}>
            <line x1={xScale(logN)} y1={PAD.top} x2={xScale(logN)} y2={H - PAD.bottom} stroke={m.color} strokeWidth={1} strokeDasharray="2 4" opacity={0.4} />
            <text x={xScale(logN)} y={PAD.top - 4} textAnchor="middle" fontSize={9} fontWeight={600} fill={m.color}>
              {m.name}
            </text>
          </g>
        );
      })}

      {/* 曲线 */}
      {EMERGENT_TASKS.map((task, i) => {
        const pts = task.curve.map((p) => `${xScale(p.logN)},${yScale(p.acc)}`).join(" ");
        return (
          <g key={task.name}>
            <polyline fill="none" stroke={COLORS[i]} strokeWidth={2.4} points={pts} />
            {task.curve.map((p, k) => (
              <circle key={k} cx={xScale(p.logN)} cy={yScale(p.acc)} r={3} fill={COLORS[i]} />
            ))}
          </g>
        );
      })}

      {/* 图例 */}
      <g transform={`translate(${PAD.left + 8}, ${PAD.top + 12})`}>
        {EMERGENT_TASKS.map((task, i) => (
          <g key={task.name} transform={`translate(0, ${i * 16})`}>
            <line x1={0} x2={14} y1={0} y2={0} stroke={COLORS[i]} strokeWidth={2.4} />
            <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">{task.name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
