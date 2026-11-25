import { useState } from "react";
import { generateRlTrajectory } from "../lib/data";

const W = 500;
const H = 260;

export function RlTrajectoryWidget() {
  const [steps, setSteps] = useState(10);
  const traj = generateRlTrajectory(steps);

  const xs = traj.map((p) => p.x), ys = traj.map((p) => p.y);
  const minX = Math.min(...xs, -1), maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1), maxY = Math.max(...ys, 1);
  const toX = (x: number) => 30 + ((x - minX) / (maxX - minX || 1)) * (W - 60);
  const toY = (y: number) => 30 + ((y - minY) / (maxY - minY || 1)) * (H - 60);
  const path = traj.map((p, i) => `${i === 0 ? "M" : "L"} ${toX(p.x)} ${toY(p.y)}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        轨迹步数 = {steps}
        <input type="range" min={2} max={30} value={steps} onChange={(e) => setSteps(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`RL agent 自动生成的 ${steps} 步训练轨迹`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          RL agent 自我博弈生成的训练轨迹(替代人类录屏)
        </text>
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
        {traj.length > 0 && <circle cx={toX(traj[traj.length - 1].x)} cy={toY(traj[traj.length - 1].y)} r={5} fill="#9d174d" />}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        最新动作:{traj[traj.length - 1]?.action ?? "-"}。整段(帧,动作)序列全部由 agent 自我博弈自动产生,不依赖任何人类玩家录屏标注。
      </p>
    </div>
  );
}
