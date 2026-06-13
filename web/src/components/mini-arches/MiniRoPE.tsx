import type { MiniArchProps } from "./types";

export function MiniRoPE({
  width = 160,
  height = 70,
  ariaLabel = "RoPE 架构缩图",
}: MiniArchProps) {
  // 4 个时刻的 query 向量按不同角度旋转 — 体现"位置变成旋转角度"
  // 每个时刻一个圆 + 一根指针,指针角度随位置增长
  const positions = [12, 52, 92, 132];
  const center_y = 35;
  const r = 14;
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {positions.map((cx, i) => {
          // 角度按位置线性增长 — RoPE 的 mθ 旋转
          const angleDeg = -30 - i * 60; // 起始 -30,每步 -60
          const angleRad = (angleDeg * Math.PI) / 180;
          const tipX = cx + r * Math.cos(angleRad);
          const tipY = center_y + r * Math.sin(angleRad);
          return (
            <g key={i}>
              {/* 圆盘表示 2D 旋转平面 */}
              <circle
                cx={cx}
                cy={center_y}
                r={r}
                className="illustration__block illustration__block--alt"
                fill="none"
              />
              {/* 指针 — query 向量经过 R_m 旋转后的方向 */}
              <line
                x1={cx}
                y1={center_y}
                x2={tipX}
                y2={tipY}
                className="illustration__branch illustration__branch--v"
              />
              {/* 圆心点 */}
              <circle
                cx={cx}
                cy={center_y}
                r={1.5}
                className="illustration__addnorm"
              />
              {/* 位置标签 */}
              <text
                x={cx}
                y={62}
                textAnchor="middle"
                fontSize="7"
                fill="currentColor"
                opacity="0.6"
              >
                m={i}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
