import { DATA_POINT, NOISE_POINT } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  mode: "fm" | "ddpm";
  t: number; // 0(纯噪声) → 1(数据),用于标出当前 x_t 在路径上的位置
}

const GRID_X = [140, 260, 380, 500];
const GRID_Y = [90, 170, 250];

function normalize(dx: number, dy: number, len = 34) {
  const mag = Math.hypot(dx, dy) || 1;
  return { dx: (dx / mag) * len, dy: (dy / mag) * len };
}

export function VelocityFieldDiagram({ mode, t }: Props) {
  // Flow Matching: 速度场处处等于 x_1 - x_0 的常数方向(整条直线上不变)
  const constDir = normalize(NOISE_POINT.x - DATA_POINT.x, NOISE_POINT.y - DATA_POINT.y);

  // 沿直线路径的当前采样点 x_t = (1-t)·x0 + t·x1(t=1 是噪声,t=0 是数据 —— 与源文档 sample() 的反向积分方向一致)
  const curX = (1 - t) * DATA_POINT.x + t * NOISE_POINT.x;
  const curY = (1 - t) * DATA_POINT.y + t * NOISE_POINT.y;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={mode === "fm" ? "Flow Matching 速度场(处处恒定)" : "DDPM 瞬时切线场(随位置变化)"}
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "fm"
          ? "Flow Matching — v(x_t, t) = x₁ - x₀,整条直线方向恒定"
          : "DDPM — 瞬时切线方向随位置 / schedule 变化,任务更复杂"}
      </text>

      {GRID_X.map((gx) =>
        GRID_Y.map((gy) => {
          const dir = mode === "fm" ? constDir : normalize(NOISE_POINT.x - gx, NOISE_POINT.y - gy - (gx - 320) * 0.35);
          const x2 = gx + dir.dx;
          const y2 = gy + dir.dy;
          const angle = (Math.atan2(y2 - gy, x2 - gx) * 180) / Math.PI;
          return (
            <g key={`${gx}-${gy}`}>
              <line x1={gx} y1={gy} x2={x2} y2={y2} stroke={mode === "fm" ? "#3b82f6" : "#ec4899"} strokeWidth={2} />
              <polygon
                points="0,-4 8,0 0,4"
                fill={mode === "fm" ? "#3b82f6" : "#ec4899"}
                transform={`translate(${x2}, ${y2}) rotate(${angle})`}
              />
            </g>
          );
        }),
      )}

      <line x1={NOISE_POINT.x} y1={NOISE_POINT.y} x2={DATA_POINT.x} y2={DATA_POINT.y} stroke="var(--border)" strokeWidth={1} strokeDasharray="3 3" />

      <circle cx={NOISE_POINT.x} cy={NOISE_POINT.y} r={9} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.8} />
      <text x={NOISE_POINT.x + 16} y={NOISE_POINT.y + 4} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        {NOISE_POINT.label}
      </text>

      <circle cx={DATA_POINT.x} cy={DATA_POINT.y} r={9} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.8} />
      <text x={DATA_POINT.x - 16} y={DATA_POINT.y + 22} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        {DATA_POINT.label}
      </text>

      <circle cx={curX} cy={curY} r={7} fill="#fce7f3" stroke="#db2777" strokeWidth={2} />
      <text x={curX} y={curY - 14} textAnchor="middle" fontSize={10} fontWeight={700} fill="#db2777">
        x_t (t={t.toFixed(2)})
      </text>

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10.5} fill="var(--ink-muted)">
        {mode === "fm"
          ? "网络只需学「这条直线的方向」—— 比学「曲线上每点的瞬时切线」简单得多"
          : "曲线路径上,速度方向随 x_t 位置 / t 而变,回归目标更复杂"}
      </text>
    </svg>
  );
}
