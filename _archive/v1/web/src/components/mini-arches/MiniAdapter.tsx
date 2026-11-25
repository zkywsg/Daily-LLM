import type { MiniArchProps } from "./types";

export function MiniAdapter({
  width = 160,
  height = 70,
  ariaLabel = "Adapter Tuning 架构缩图",
}: MiniArchProps) {
  // Transformer layer 里插入 bottleneck adapter,base 冻结
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* x */}
        <rect x="2" y="30" width="12" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="8" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">x</text>
        {/* Frozen attention(灰色虚线 = 冻结) */}
        <rect x="20" y="14" width="26" height="42" rx="2"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__block" opacity="0.6" />
        <text x="33" y="22" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.65">Attn</text>
        <text x="33" y="29" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.55">❄ frozen</text>
        <text x="33" y="40" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.65">FFN</text>
        <text x="33" y="47" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.55">❄ frozen</text>
        {/* Adapter bottleneck — 大图突出 down-up 结构 */}
        <g transform="translate(56, 8)">
          <text x="22" y="3" textAnchor="middle" fontSize="4.5" fontWeight="600"
            fill="currentColor" opacity="0.85">adapter (trainable)</text>
          {/* down 块(宽,顶部) */}
          <rect x="0" y="8" width="44" height="8" rx="1.5"
            className="illustration__featuremap illustration__featuremap--ctx" />
          <text x="22" y="14" textAnchor="middle" fontSize="4.5"
            fill="currentColor" opacity="0.85">down d→r</text>
          {/* bottleneck — 窄长方形 */}
          <rect x="14" y="20" width="16" height="6" rx="1"
            className="illustration__featuremap illustration__featuremap--ctx" opacity="0.8" />
          <text x="22" y="25" textAnchor="middle" fontSize="3.8"
            fill="currentColor" opacity="0.8">r=64</text>
          {/* ReLU */}
          <text x="22" y="33" textAnchor="middle" fontSize="4"
            fill="currentColor" opacity="0.7">ReLU</text>
          {/* up 块 */}
          <rect x="0" y="36" width="44" height="8" rx="1.5"
            className="illustration__featuremap illustration__featuremap--ctx" />
          <text x="22" y="42" textAnchor="middle" fontSize="4.5"
            fill="currentColor" opacity="0.85">up r→d</text>
          {/* + residual */}
          <text x="22" y="52" textAnchor="middle" fontSize="5" fontWeight="600"
            fill="currentColor" opacity="0.85">+ x</text>
        </g>
        {/* 输出 y */}
        <rect x="118" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="125" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">y</text>
        {/* 箭头 */}
        <line x1="14" y1="35" x2="20" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="46" y1="35" x2="56" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="100" y1="35" x2="118" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="66" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          只训 3% 参数 · 推理稍慢
        </text>
      </g>
    </svg>
  );
}
