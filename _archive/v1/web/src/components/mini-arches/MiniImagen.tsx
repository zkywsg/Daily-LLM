import type { MiniArchProps } from "./types";

export function MiniImagen({
  width = 160,
  height = 70,
  ariaLabel = "Imagen / CFG 架构缩图",
}: MiniArchProps) {
  // T5-XXL 大块文本编码器 + 3 级 cascade 逐渐放大的图像方块
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 左侧:T5-XXL 大文本编码器(粗块强调"巨大") */}
        <rect x="2" y="14" width="20" height="42" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="12" y="33" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">T5</text>
        <text x="12" y="40" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">XXL</text>
        <text x="12" y="51" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">11B</text>
        {/* 3 级 cascade 图像方块 — 逐渐变大 */}
        {[
          { x: 36, size: 10, label: "64²", color: "ffn" },
          { x: 64, size: 18, label: "256²", color: "ffn" },
          { x: 98, size: 26, label: "1024²", color: "ctx" },
        ].map((s, i) => {
          const cy = 32;
          const y = cy - s.size / 2;
          const cls = s.color === "ctx"
            ? "illustration__featuremap illustration__featuremap--ctx"
            : "illustration__proj illustration__proj--ffn";
          return (
            <g key={i}>
              <rect x={s.x} y={y} width={s.size} height={s.size} rx="1.5"
                className={cls} />
              <text x={s.x + s.size / 2} y={cy + s.size / 2 + 8}
                textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.65">
                {s.label}
              </text>
            </g>
          );
        })}
        {/* T5 文本注入到每级 cascade(虚线) */}
        {[41, 73, 111].map((tx, i) => (
          <line key={i} x1="22" y1="32" x2={tx} y2="32"
            className="illustration__residual" />
        ))}
        {/* cascade 间放大箭头 */}
        <line x1="46" y1="32" x2="64" y2="32"
          className="illustration__branch illustration__branch--q" />
        <line x1="82" y1="32" x2="98" y2="32"
          className="illustration__branch illustration__branch--q" />
        {/* 底部:CFG */}
        <text x="80" y="65" textAnchor="middle" fontSize="5" fill="currentColor" opacity="0.55">
          T5-XXL + CFG · 3 级 cascade
        </text>
      </g>
    </svg>
  );
}
