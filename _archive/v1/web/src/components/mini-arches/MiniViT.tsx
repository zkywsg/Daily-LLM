import type { MiniArchProps } from "./types";

export function MiniViT({
  width = 160,
  height = 70,
  ariaLabel = "ViT 架构缩图",
}: MiniArchProps) {
  // 图像 → 切 patch 网格 → Transformer encoder → [CLS] 输出
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:图像 + patch 切分网格 */}
        <rect
          x="4"
          y="14"
          width="40"
          height="40"
          rx="2"
          fill="none"
          className="illustration__layer illustration__layer--input"
        />
        {/* 4×4 patch 网格线 */}
        {[14, 24, 34, 44, 54].map((y) => (
          <line key={`h-${y}`} x1="4" y1={y} x2="44" y2={y} stroke="currentColor" strokeWidth="0.5" opacity="0.4" />
        ))}
        {[4, 14, 24, 34, 44].map((x) => (
          <line key={`v-${x}`} x1={x} y1="14" x2={x} y2="54" stroke="currentColor" strokeWidth="0.5" opacity="0.4" />
        ))}
        {/* 中央:Transformer encoder stack */}
        {Array.from({ length: 10 }, (_, i) => (
          <rect
            key={i}
            x="64"
            y={14 + i * 4}
            width="50"
            height="2.8"
            rx="0.5"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* patches → encoder 箭头 */}
        <line
          x1="44"
          y1="34"
          x2="64"
          y2="34"
          className="illustration__branch illustration__branch--q"
        />
        {/* 右侧:[CLS] → class 输出 */}
        <rect
          x="128"
          y="30"
          width="28"
          height="10"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <line
          x1="114"
          y1="34"
          x2="128"
          y2="34"
          className="illustration__branch illustration__branch--v"
        />
        {/* 底部标 "patch 16×16" */}
        <text
          x="24"
          y="64"
          textAnchor="middle"
          fontSize="6"
          fill="currentColor"
          opacity="0.6"
        >
          16×16 patches
        </text>
      </g>
    </svg>
  );
}
