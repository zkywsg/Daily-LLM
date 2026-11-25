import type { MiniArchProps } from "./types";

export function MiniSwitch({
  width = 160,
  height = 70,
  ariaLabel = "Switch Transformer 架构缩图",
}: MiniArchProps) {
  // Top-1:只激活 1 个 expert,极简路由
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
        <rect x="2" y="30" width="14" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="9" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">x</text>
        {/* Switch */}
        <rect x="22" y="22" width="22" height="26" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="33" y="31" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">Switch</text>
        <text x="33" y="39" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">top-1</text>
        <text x="33" y="45" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.65">argmax</text>
        {/* 8 个 expert,只 1 个高亮 */}
        {[
          { y: 6, active: false },
          { y: 14, active: false },
          { y: 22, active: false },
          { y: 30, active: true },  // 唯一高亮
          { y: 38, active: false },
          { y: 46, active: false },
          { y: 54, active: false },
          { y: 62, active: false },
        ].map((e, i) => (
          e.active
            ? <rect key={i} x="52" y={e.y} width="40" height="6" rx="1"
                className="illustration__featuremap illustration__featuremap--ctx" />
            : <rect key={i} x="52" y={e.y} width="40" height="6" rx="1"
                className="illustration__proj illustration__proj--ffn" opacity="0.22" />
        ))}
        <text x="72" y="3" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">N=128 experts</text>
        {/* 单箭头到唯一选中的 expert */}
        <line x1="44" y1="35" x2="52" y2="33"
          className="illustration__branch illustration__branch--q"
          strokeWidth="1.2" />
        {/* 选中 expert 到输出 */}
        <line x1="92" y1="33" x2="132" y2="35"
          className="illustration__branch illustration__branch--v"
          strokeWidth="1.2" />
        {/* y */}
        <rect x="132" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="139" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">y</text>
        {/* 底部 */}
        <text x="80" y="69" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          每 token 走 1 个 expert · 1.6T 总参 · 4× 提速
        </text>
      </g>
    </svg>
  );
}
