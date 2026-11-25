import type { MiniArchProps } from "./types";

export function MiniDDPM({
  width = 160,
  height = 70,
  ariaLabel = "DDPM 架构缩图",
}: MiniArchProps) {
  // 5 个方块从纯噪声 → 清晰图,逐步去噪;底部箭头标 timestep T → 0
  const steps = [
    { x: 4, noise: 1.0 },    // x_T 纯噪声
    { x: 36, noise: 0.75 },  // x_t
    { x: 68, noise: 0.5 },
    { x: 100, noise: 0.25 },
    { x: 132, noise: 0.0 },  // x_0 清晰
  ];
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {steps.map((s, i) => (
          <g key={i}>
            {/* 主方块 — 从噪声色到清晰色渐变 */}
            <rect
              x={s.x}
              y="14"
              width="24"
              height="24"
              rx="2"
              className={
                s.noise > 0.6
                  ? "illustration__proj illustration__proj--ffn"
                  : s.noise > 0.2
                  ? "illustration__proj illustration__proj--v"
                  : "illustration__featuremap illustration__featuremap--ctx"
              }
              opacity={1 - s.noise * 0.3}
            />
            {/* 噪点撒点(高 noise 时多,低 noise 时少) */}
            {Array.from({ length: Math.round(s.noise * 8) }, (_, j) => (
              <circle
                key={j}
                cx={s.x + 4 + (j % 4) * 5}
                cy={18 + Math.floor(j / 4) * 5}
                r="0.8"
                fill="currentColor"
                opacity="0.35"
              />
            ))}
          </g>
        ))}
        {/* 反向箭头连接 — 从右到左 */}
        {[0, 1, 2, 3].map((i) => (
          <line
            key={i}
            x1={steps[i].x + 24}
            y1="26"
            x2={steps[i + 1].x}
            y2="26"
            className="illustration__branch illustration__branch--v"
          />
        ))}
        {/* 底部 timestep 标 */}
        <text x="16" y="50" textAnchor="middle" fontSize="6" fill="currentColor" opacity="0.7">x_T</text>
        <text x="144" y="50" textAnchor="middle" fontSize="6" fill="currentColor" opacity="0.7">x_0</text>
        <text x="80" y="64" textAnchor="middle" fontSize="6" fill="currentColor" opacity="0.55">
          1000 步去噪
        </text>
      </g>
    </svg>
  );
}
