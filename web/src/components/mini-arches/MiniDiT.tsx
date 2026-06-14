import type { MiniArchProps } from "./types";

export function MiniDiT({
  width = 160,
  height = 70,
  ariaLabel = "DiT 架构缩图",
}: MiniArchProps) {
  // Noisy latent patch → Transformer with AdaLN conditioning → predicted noise
  // 关键视觉:左侧带噪 patch 网格 + 顶部 timestep/class 条件流入
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:噪声 latent patch 网格 */}
        <rect x="4" y="22" width="28" height="28" rx="2" fill="none"
          className="illustration__layer illustration__layer--input" />
        {/* 模拟 noisy 模式:几个不规则填充小方块 */}
        {[
          {x: 4, y: 22, fill: 0.3}, {x: 11, y: 22, fill: 0.6},
          {x: 18, y: 22, fill: 0.2}, {x: 25, y: 22, fill: 0.8},
          {x: 4, y: 29, fill: 0.7}, {x: 11, y: 29, fill: 0.3},
          {x: 18, y: 29, fill: 0.5}, {x: 25, y: 29, fill: 0.2},
          {x: 4, y: 36, fill: 0.4}, {x: 11, y: 36, fill: 0.7},
          {x: 18, y: 36, fill: 0.3}, {x: 25, y: 36, fill: 0.6},
          {x: 4, y: 43, fill: 0.5}, {x: 11, y: 43, fill: 0.4},
          {x: 18, y: 43, fill: 0.8}, {x: 25, y: 43, fill: 0.3},
        ].map((c, i) => (
          <rect key={i} x={c.x} y={c.y} width="7" height="7"
            className="illustration__proj illustration__proj--ffn" opacity={c.fill} />
        ))}
        {/* 顶部条件:t 和 c */}
        <rect x="60" y="2" width="14" height="8" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="67" y="8" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.75">t</text>
        <rect x="80" y="2" width="14" height="8" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="87" y="8" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.75">c</text>
        {/* 中央:DiT block stack — 注意每层都被条件调制(虚线点缀) */}
        {Array.from({ length: 10 }, (_, i) => (
          <g key={i}>
            <rect x="52" y={16 + i * 3.4} width="54" height="2.4" rx="0.5"
              className="illustration__proj illustration__proj--v" />
            <circle cx="46" cy={17.2 + i * 3.4} r="1.2"
              className="illustration__addnorm" />
          </g>
        ))}
        {/* 顶部 (t,c) → AdaLN-Zero 调制线 */}
        <line x1="80" y1="10" x2="48" y2="18"
          className="illustration__residual" />
        {/* 左侧 noisy → encoder */}
        <line x1="32" y1="36" x2="52" y2="36"
          className="illustration__branch illustration__branch--q" />
        {/* 右侧:预测噪声 */}
        <rect x="118" y="32" width="38" height="8" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="137" y="38" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.75">ε̂</text>
        <line x1="106" y1="36" x2="118" y2="36"
          className="illustration__branch illustration__branch--v" />
        {/* 底部标 AdaLN-Zero */}
        <text x="80" y="66" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.55">AdaLN-Zero 条件注入</text>
      </g>
    </svg>
  );
}
