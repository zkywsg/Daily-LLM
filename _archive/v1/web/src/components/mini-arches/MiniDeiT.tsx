import type { MiniArchProps } from "./types";

export function MiniDeiT({
  width = 160,
  height = 70,
  ariaLabel = "DeiT 架构缩图",
}: MiniArchProps) {
  // ViT 形态 + 额外的 distill token + CNN teacher 蒸馏箭头
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左下:图像 patch 网格(缩小) */}
        <rect
          x="4"
          y="36"
          width="24"
          height="24"
          rx="2"
          fill="none"
          className="illustration__layer illustration__layer--input"
        />
        {[36, 42, 48, 54, 60].map((y) => (
          <line key={`h-${y}`} x1="4" y1={y} x2="28" y2={y} stroke="currentColor" strokeWidth="0.4" opacity="0.4" />
        ))}
        {[4, 10, 16, 22, 28].map((x) => (
          <line key={`v-${x}`} x1={x} y1="36" x2={x} y2="60" stroke="currentColor" strokeWidth="0.4" opacity="0.4" />
        ))}
        {/* 左上:CNN teacher 块 */}
        <rect
          x="4"
          y="8"
          width="24"
          height="20"
          rx="2"
          className="illustration__proj illustration__proj--act"
        />
        <text x="16" y="20" textAnchor="middle" fontSize="6" fontWeight="600" fill="currentColor" opacity="0.7">CNN</text>
        <text x="16" y="26" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.7">teacher</text>
        {/* 中央:ViT encoder */}
        {Array.from({ length: 10 }, (_, i) => (
          <rect
            key={i}
            x="52"
            y={14 + i * 4}
            width="50"
            height="2.8"
            rx="0.5"
            className="illustration__proj illustration__proj--ffn"
          />
        ))}
        {/* patches + CNN teacher → encoder 两个箭头 */}
        <line x1="28" y1="48" x2="52" y2="48" className="illustration__branch illustration__branch--q" />
        <line x1="28" y1="18" x2="52" y2="22" className="illustration__branch illustration__branch--v" />
        {/* 右侧 双 head: [CLS] 和 [Distill] */}
        <rect
          x="118"
          y="18"
          width="36"
          height="10"
          rx="1"
          className="illustration__featuremap illustration__featuremap--ctx"
        />
        <text x="136" y="25" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.7">CLS</text>
        <rect
          x="118"
          y="40"
          width="36"
          height="10"
          rx="1"
          className="illustration__proj illustration__proj--v"
        />
        <text x="136" y="47" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.7">distill</text>
        <line x1="102" y1="22" x2="118" y2="22" className="illustration__branch illustration__branch--v" />
        <line x1="102" y1="44" x2="118" y2="44" className="illustration__branch illustration__branch--v" />
      </g>
    </svg>
  );
}
