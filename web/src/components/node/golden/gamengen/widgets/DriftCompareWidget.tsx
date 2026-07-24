import { useState } from "react";
import { driftCurve } from "../lib/data";

const W = 600;
const H = 280;

// 开关"是否对条件帧加噪声增强",对比长程自回归多步生成后画面质量
// 是否发生漂移退化。这是 GameNGen 论文的核心工程细节。

export function DriftCompareWidget() {
  const [withAug, setWithAug] = useState(false);
  const steps = 60;
  const curveOff = driftCurve(false, steps);
  const curveOn = driftCurve(true, steps);

  const toX = (t: number) => 40 + (t / steps) * (W - 80);
  const toY = (v: number) => H - 40 - v * (H - 80);
  const pathOff = curveOff.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");
  const pathOn = curveOn.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");

  return (
    <div>
      <button
        type="button" onClick={() => setWithAug((v) => !v)} aria-pressed={withAug}
        style={{
          padding: "4px 14px", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)",
          border: `1px solid ${withAug ? "#d946ef" : "var(--border)"}`,
          background: withAug ? "#d946ef" : "var(--bg-surface)",
          color: withAug ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
        }}
      >
        {withAug ? "✓ 已开启噪声增强" : "开启噪声增强"}
      </button>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="噪声增强对长程自回归漂移的影响对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          长程自回归 {steps} 步后的画面质量衰减(漂移)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <path d={pathOff} fill="none" stroke="#9ca3af" strokeWidth={2} strokeDasharray={withAug ? "3 3" : undefined} opacity={withAug ? 0.4 : 1} />
        <path d={pathOn} fill="none" stroke="#d946ef" strokeWidth={2} strokeDasharray={withAug ? undefined : "3 3"} opacity={withAug ? 1 : 0.4} />
        <text x={W - 45} y={toY(curveOff[curveOff.length - 1]) + 4} textAnchor="end" fontSize={10} fill="#9ca3af">无增强</text>
        <text x={W - 45} y={toY(curveOn[curveOn.length - 1]) - 6} textAnchor="end" fontSize={10} fill="#d946ef">有增强</text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        不加噪声增强:模型训练时只见过"完美"的条件帧,推理时自己生成的略有瑕疵的帧作为下一步条件会导致误差越滚越大(灰线快速下滑)。加噪声增强后模型学会了容忍不完美条件帧,漂移显著减缓(粉线)。
      </p>
    </div>
  );
}
