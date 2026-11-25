import type { MiniArchProps } from "./types";

export function MiniFlowMatching({
  width = 160,
  height = 70,
  ariaLabel = "Flow Matching 架构缩图",
}: MiniArchProps) {
  // 左侧噪声 x_1 + 右侧数据 x_0 中间一条直线 + 速度箭头
  // 体现"在 noise 和 data 间直线插值,学速度场"
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:x_1(噪声) */}
        <circle cx="20" cy="35" r="10" className="illustration__proj illustration__proj--ffn" />
        <text x="20" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">x_1</text>
        <text x="20" y="54" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.65">noise</text>
        {/* 右侧:x_0(数据) */}
        <circle cx="140" cy="35" r="10" className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="140" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">x_0</text>
        <text x="140" y="54" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.65">data</text>
        {/* 中央:直线插值路径 */}
        <line x1="30" y1="35" x2="130" y2="35"
          stroke="currentColor" strokeWidth="1.5" opacity="0.5" strokeDasharray="3,2" />
        {/* 中间点 x_t */}
        <circle cx="80" cy="35" r="5" className="illustration__addnorm" />
        <text x="80" y="22" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">x_t</text>
        {/* 速度场箭头 — 沿直线指向 x_0 */}
        {[50, 65, 95, 110].map((x, i) => (
          <g key={i}>
            <line x1={x} y1="35" x2={x + 8} y2="35"
              className="illustration__branch illustration__branch--v" />
            <path d={`M ${x + 8} 35 L ${x + 5} 33 L ${x + 5} 37 Z`}
              fill="currentColor" opacity="0.7" />
          </g>
        ))}
        {/* 顶部速度公式 */}
        <text x="80" y="10" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.75">v = x_1 - x_0</text>
        {/* 底部"直线路径"标 */}
        <text x="80" y="66" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">直线插值 · ODE 反向</text>
      </g>
    </svg>
  );
}
