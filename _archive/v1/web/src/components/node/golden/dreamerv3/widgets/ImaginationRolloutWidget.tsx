import { useState } from "react";
import { imaginationRollout } from "../lib/data";

const W = 500;
const H = 300;

export function ImaginationRolloutWidget() {
  const [steps, setSteps] = useState(5);
  const traj = imaginationRollout(steps);

  const xs = traj.map((p) => p.x);
  const ys = traj.map((p) => p.y);
  const minX = Math.min(...xs, -1), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1), maxY = Math.max(...ys, 1);
  const toX = (x: number) => 30 + ((x - minX) / (maxX - minX)) * (W - 60);
  const toY = (y: number) => 30 + ((y - minY) / (maxY - minY)) * (H - 60);

  const path = traj.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.x)} ${toY(p.y)}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        想象步数 = {steps}
        <input type="range" min={1} max={15} value={steps} onChange={(e) => setSteps(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`纯想象轨迹,${steps} 步`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          actor-critic 训练用的纯想象轨迹(不接触真实环境)
        </text>
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} strokeDasharray="4 2" />
        {traj.map((p, i) => (
          <circle key={i} cx={toX(p.x)} cy={toY(p.y)} r={i === 0 ? 6 : 4} fill={i === 0 ? "#9d174d" : "#d946ef"} />
        ))}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        深色起点是当前真实状态编码,之后每一步都由世界模型在隐空间里自回归展开——actor 和 critic 的梯度全部来自这条虚拟轨迹。
      </p>
    </div>
  );
}
