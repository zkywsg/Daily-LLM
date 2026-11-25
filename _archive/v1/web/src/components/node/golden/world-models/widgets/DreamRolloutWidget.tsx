import { useState } from "react";
import { dreamRollout } from "../lib/data";

const W = 680;
const H = 260;

// 完全在 M 自回归生成的"梦境"轨迹上跑 rollout:z0 → M → z1 → M → z2 → …,
// 全程不接触真实帧/环境。点"梦境前进一步"每次追加一步。

export function DreamRolloutWidget() {
  const [steps, setSteps] = useState(0);
  const traj = dreamRollout(0.2, steps);

  const xMin = 0, xMax = 10;
  const yMin = -1.5, yMax = 1.5;
  const toX = (t: number) => 40 + (t / xMax) * (W - 80);
  const toY = (v: number) => H - 40 - ((v - yMin) / (yMax - yMin)) * (H - 80);

  const path = traj.map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i)} ${toY(v)}`).join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`梦境 rollout,已进行 ${steps} 步`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          纯梦境 rollout(z0 → M → z1 → M → z2 → …),已进行 {steps} 步
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <line x1={40} y1={30} x2={40} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} strokeDasharray="4 2" />
        {traj.map((v, i) => (
          <circle key={i} cx={toX(i)} cy={toY(v)} r={4} fill="#d946ef" />
        ))}
      </svg>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          type="button"
          onClick={() => setSteps((s) => Math.min(s + 1, 10))}
          disabled={steps >= 10}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: steps >= 10 ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: steps >= 10 ? 0.5 : 1 }}
        >
          梦境前进一步
        </button>
        <button
          type="button"
          onClick={() => setSteps(0)}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        虚线是完全由 M 自回归生成的轨迹 —— C 的策略训练全程只看这条轨迹,从未调用过真实环境或 V 编码的真实帧。
      </p>
    </div>
  );
}
