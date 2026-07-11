import { CURVED_WAYPOINTS, DATA_POINT, NOISE_POINT } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  mode: "straight" | "curved";
}

export function StraightVsCurvedPathDiagram({ mode }: Props) {
  const straightPath = `M ${NOISE_POINT.x} ${NOISE_POINT.y} L ${DATA_POINT.x} ${DATA_POINT.y}`;
  const curvedPath = `M ${CURVED_WAYPOINTS.map((p) => `${p.x} ${p.y}`).join(" L ")}`;

  const dots = mode === "straight"
    ? Array.from({ length: 6 }, (_, i) => {
        const t = i / 5;
        return {
          x: NOISE_POINT.x + (DATA_POINT.x - NOISE_POINT.x) * t,
          y: NOISE_POINT.y + (DATA_POINT.y - NOISE_POINT.y) * t,
        };
      })
    : CURVED_WAYPOINTS;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={mode === "straight" ? "Flow Matching 直线路径" : "DDPM 曲线随机路径"}
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "straight"
          ? "Flow Matching — x_t = (1-t)·x₀ + t·x₁,直线插值"
          : "DDPM — x_t = √āₜ·x₀ + √(1-āₜ)·ε,schedule 决定的曲线"}
      </text>

      <path
        d={mode === "straight" ? straightPath : curvedPath}
        fill="none"
        stroke={mode === "straight" ? "#3b82f6" : "#ec4899"}
        strokeWidth={2.5}
        strokeDasharray={mode === "straight" ? undefined : "1 0"}
      />

      {dots.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={i === 0 || i === dots.length - 1 ? 0 : 4}
          fill={mode === "straight" ? "#dbeafe" : "#fce7f3"}
          stroke={mode === "straight" ? "#3b82f6" : "#ec4899"}
          strokeWidth={1.4}
        />
      ))}

      <circle cx={NOISE_POINT.x} cy={NOISE_POINT.y} r={9} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.8} />
      <text x={NOISE_POINT.x + 16} y={NOISE_POINT.y + 4} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        {NOISE_POINT.label}
      </text>

      <circle cx={DATA_POINT.x} cy={DATA_POINT.y} r={9} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.8} />
      <text x={DATA_POINT.x - 16} y={DATA_POINT.y + 22} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        {DATA_POINT.label}
      </text>

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10.5} fill="var(--ink-muted)">
        {mode === "straight"
          ? "任意点速度恒定 v = x₁ - x₀ → 大步 ODE 积分也稳(10-20 步)"
          : "路径弯曲 + 随机噪声注入 → 需要小步 SDE 反向(50-100 步)"}
      </text>
    </svg>
  );
}
