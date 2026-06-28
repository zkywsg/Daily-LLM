import { SCALE_POINTS, fmtParams } from "../lib/data";

const W = 700;
const H = 340;
const PAD = { left: 60, right: 80, top: 36, bottom: 60 };

// CoT 在 ~100B 参数后才显著有效。
// 横轴 log 参数量,纵轴 GSM8K 准确率,两条曲线:standard / CoT。
// CoT 在 175B 处突然跳变,小模型 CoT 反而比 standard 还差(小模型推一步就跑偏)。

export function ScaleEmergenceCurve() {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const logMin = Math.log10(0.3e9);
  const logMax = Math.log10(600e9);
  const xScale = (n: number) => PAD.left + ((Math.log10(n) - logMin) / (logMax - logMin)) * innerW;

  const accMin = 0;
  const accMax = 0.7;
  const yScale = (a: number) => PAD.top + (1 - (a - accMin) / (accMax - accMin)) * innerH;

  const stdPts = SCALE_POINTS.map((p) => `${xScale(p.params)},${yScale(p.standard)}`).join(" ");
  const cotPts = SCALE_POINTS.map((p) => `${xScale(p.params)},${yScale(p.cot)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CoT emergence curve">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        CoT 涌现 — GSM8K 准确率 vs 模型参数(Wei 2022)
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* 模型刻度 */}
      {SCALE_POINTS.map((p, i) => (
        <g key={p.model}>
          <line x1={xScale(p.params)} y1={H - PAD.bottom} x2={xScale(p.params)} y2={H - PAD.bottom + 4} stroke="var(--ink-muted)" />
          <text
            x={xScale(p.params)}
            y={H - PAD.bottom + 18}
            textAnchor="middle"
            fontSize={9}
            fill="var(--ink-muted)"
          >
            {fmtParams(p.params)}
          </text>
          {i % 2 === 0 && (
            <text x={xScale(p.params)} y={H - PAD.bottom + 30} textAnchor="middle" fontSize={8} fill="var(--ink-muted)" fontStyle="italic">
              {p.model.replace("GPT-3 ", "").replace("PaLM ", "PaLM·")}
            </text>
          )}
        </g>
      ))}
      <text x={W / 2 - 20} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        参数量(log)→
      </text>

      {/* y 刻度 */}
      {[0, 0.2, 0.4, 0.6].map((a) => (
        <g key={a}>
          <line x1={PAD.left - 4} x2={W - PAD.right} y1={yScale(a)} y2={yScale(a)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 8} y={yScale(a) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {(a * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {/* Standard 曲线 */}
      <polyline fill="none" stroke="#9ca3af" strokeWidth={2.2} points={stdPts} />
      {SCALE_POINTS.map((p, i) => (
        <circle key={`s-${i}`} cx={xScale(p.params)} cy={yScale(p.standard)} r={4} fill="#9ca3af" />
      ))}

      {/* CoT 曲线 */}
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.4} points={cotPts} />
      {SCALE_POINTS.map((p, i) => (
        <circle key={`c-${i}`} cx={xScale(p.params)} cy={yScale(p.cot)} r={5} fill="#ec4899" />
      ))}

      {/* 涌现阈值标注 */}
      <line
        x1={xScale(60e9)}
        y1={PAD.top}
        x2={xScale(60e9)}
        y2={H - PAD.bottom}
        stroke="#10b981"
        strokeWidth={1.5}
        strokeDasharray="3 3"
      />
      <text x={xScale(60e9)} y={PAD.top - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#10b981">
        ~62B 涌现阈值
      </text>

      {/* 图例 */}
      <g transform={`translate(${W - PAD.right + 4}, ${PAD.top + 14})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#ec4899">CoT</text>
        </g>
        <g transform="translate(0, 16)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#9ca3af" strokeWidth={2.2} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#6b7280">Standard</text>
        </g>
      </g>
    </svg>
  );
}
