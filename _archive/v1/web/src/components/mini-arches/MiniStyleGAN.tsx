import type { MiniArchProps } from "./types";

export function MiniStyleGAN({
  width = 160,
  height = 70,
  ariaLabel = "StyleGAN 架构缩图",
}: MiniArchProps) {
  // z → MLP → w,常量输入逐层 AdaIN 注入 + noise
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* z */}
        <rect x="2" y="6" width="14" height="10" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="9" y="14" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.8">z</text>
        {/* Mapping MLP 8 层 */}
        <rect x="20" y="4" width="32" height="14" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="36" y="11" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">MLP × 8</text>
        <text x="36" y="16" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">mapping</text>
        {/* w */}
        <rect x="56" y="6" width="12" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="62" y="14" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">w</text>
        <line x1="16" y1="11" x2="20" y2="11"
          className="illustration__branch illustration__branch--q" />
        <line x1="52" y1="11" x2="56" y2="11"
          className="illustration__branch illustration__branch--v" />
        {/* 主路:Constant → Style Blocks 金字塔(每层 AdaIN+noise) */}
        <text x="6" y="26" fontSize="3.8" fontWeight="600"
          fill="currentColor" opacity="0.65">const</text>
        <rect x="4" y="28" width="10" height="10" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.6" />
        <text x="9" y="34" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.8">4²</text>
        {/* 4 个 style block:逐渐增大 */}
        {[
          { x: 22, w: 12, h: 14, label: "8²", coarse: "粗" },
          { x: 42, w: 16, h: 20, label: "32²", coarse: "中" },
          { x: 66, w: 22, h: 28, label: "128²", coarse: "细" },
          { x: 96, w: 32, h: 40, label: "1024²", coarse: "细" },
        ].map((b, i) => (
          <g key={i}>
            <rect x={b.x} y={50 - b.h / 2} width={b.w} height={b.h} rx="1"
              className="illustration__featuremap illustration__featuremap--ctx" />
            <text x={b.x + b.w / 2} y={62} textAnchor="middle" fontSize="3.5"
              fill="currentColor" opacity="0.6">{b.label}</text>
          </g>
        ))}
        {/* w 注入每层的虚线箭头 */}
        {[28, 50, 77, 112].map((x, i) => (
          <line key={i} x1="62" y1="16" x2={x} y2={42}
            strokeDasharray="2 2"
            className="illustration__branch illustration__branch--q" opacity="0.65" />
        ))}
        <text x="80" y="22" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">AdaIN(w) + noise per layer</text>
        {/* 主路箭头 */}
        <line x1="14" y1="33" x2="22" y2="42"
          className="illustration__branch illustration__branch--q" />
        <line x1="34" y1="46" x2="42" y2="46"
          className="illustration__branch illustration__branch--q" />
        <line x1="58" y1="46" x2="66" y2="46"
          className="illustration__branch illustration__branch--q" />
        <line x1="88" y1="46" x2="96" y2="46"
          className="illustration__branch illustration__branch--q" />
        {/* 输出 image */}
        <rect x="132" y="42" width="24" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="144" y="50" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">image</text>
        <text x="144" y="55" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">1024²</text>
        <line x1="128" y1="49" x2="132" y2="49"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="69" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.55">
          style 分层控制 · 粗→中→细 · this person does not exist
        </text>
      </g>
    </svg>
  );
}
