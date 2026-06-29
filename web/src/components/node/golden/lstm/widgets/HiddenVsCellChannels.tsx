import { useMemo } from "react";
import { simulateSequence, DEMO_SCENARIOS } from "../lib/lstm";

interface Props {
  scenarioIdx: number;
}

const W = 700;
const H = 320;
const PAD = { left: 50, right: 80, top: 36, bottom: 60 };

// 跑 N=20 步,把 h_t 和 C_t 两条曲线叠加显示。
// 让 viewer 看到 \"h 短期波动 / C 长程累积\" 的差异。

export function HiddenVsCellChannels({ scenarioIdx }: Props) {
  const scenario = DEMO_SCENARIOS[scenarioIdx];
  const STEPS = 20;
  const { hHistory, cHistory, fHistory } = useMemo(
    () => simulateSequence(STEPS, scenario.fSeq, scenario.iSeq, scenario.oSeq, scenario.gSeq),
    [scenarioIdx, scenario.fSeq, scenario.iSeq, scenario.oSeq, scenario.gSeq],
  );

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const xScale = (t: number) => PAD.left + (t / Math.max(1, STEPS - 1)) * innerW;

  // y ∈ [-2, 2]
  const yMin = -2;
  const yMax = 2;
  const yScale = (v: number) => PAD.top + (1 - (v - yMin) / (yMax - yMin)) * innerH;

  const hPts = hHistory.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
  const cPts = cHistory.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Hidden vs Cell channels: ${scenario.label}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        h_t (短期) vs C_t (长程) — 场景:{scenario.label}
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={yScale(0)} x2={W - PAD.right} y2={yScale(0)} stroke="var(--ink-muted)" strokeDasharray="2 4" />

      {/* y 刻度 */}
      {[-2, -1, 0, 1, 2].map((v) => (
        <text key={v} x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
          {v}
        </text>
      ))}

      {/* x 刻度 */}
      {[0, 5, 10, 15, 19].map((t) => (
        <text key={t} x={xScale(t)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          {t}
        </text>
      ))}
      <text x={W / 2 - 30} y={H - 30} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        timestep →
      </text>

      {/* forget gate 灰底条(显示某步 f 接近 0 → 清零事件) */}
      {fHistory.map((f, t) => {
        if (f > 0.3) return null;
        return (
          <rect
            key={`fz-${t}`}
            x={xScale(t) - 8}
            y={PAD.top}
            width={16}
            height={innerH}
            fill="#fef2f2"
            opacity={0.6}
          />
        );
      })}

      {/* h_t 曲线 */}
      <polyline fill="none" stroke="#10b981" strokeWidth={2.2} points={hPts} />
      {/* C_t 曲线 */}
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.4} points={cPts} />

      {/* 图例 */}
      <g transform={`translate(${W - PAD.right - 4}, ${PAD.top + 14})`}>
        <g transform="translate(-50, 0)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#ec4899">C_t</text>
        </g>
        <g transform="translate(-50, 16)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#10b981" strokeWidth={2.2} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#10b981">h_t</text>
        </g>
      </g>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {scenario.desc}
      </text>
    </svg>
  );
}
