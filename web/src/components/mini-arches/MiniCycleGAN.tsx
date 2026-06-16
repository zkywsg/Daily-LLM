import type { MiniArchProps } from "./types";

export function MiniCycleGAN({
  width = 160,
  height = 70,
  ariaLabel = "CycleGAN 架构缩图",
}: MiniArchProps) {
  // X ⇄ Y 双向,cycle consistency 弧线
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* X 域(左) */}
        <rect x="6" y="28" width="22" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="17" y="34" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">X</text>
        <text x="17" y="40" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">(horse)</text>
        {/* G: X→Y */}
        <rect x="40" y="6" width="22" height="16" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="51" y="14" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">G</text>
        <text x="51" y="20" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.75">X→Y</text>
        {/* F: Y→X */}
        <rect x="40" y="48" width="22" height="16" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="51" y="56" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">F</text>
        <text x="51" y="62" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.75">Y→X</text>
        {/* Y 域(中) */}
        <rect x="74" y="28" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="85" y="34" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">Y</text>
        <text x="85" y="40" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">(zebra)</text>
        {/* X → G → Y */}
        <line x1="28" y1="33" x2="40" y2="14"
          className="illustration__branch illustration__branch--q" />
        <line x1="62" y1="14" x2="78" y2="32"
          className="illustration__branch illustration__branch--q" />
        {/* Y → F → X */}
        <line x1="74" y1="38" x2="62" y2="56"
          className="illustration__branch illustration__branch--v" />
        <line x1="40" y1="56" x2="28" y2="38"
          className="illustration__branch illustration__branch--v" />
        {/* Cycle consistency 弧线(虚线) */}
        <path d="M 17 28 Q 50 -2 85 28"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--q" opacity="0.5" />
        <path d="M 17 42 Q 50 72 85 42"
          fill="none"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--v" opacity="0.5" />
        {/* 右侧:F(G(x)) ≈ x */}
        <rect x="108" y="14" width="48" height="20" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="132" y="22" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.8">cycle loss</text>
        <text x="132" y="29" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">F(G(x)) ≈ x</text>
        <rect x="108" y="38" width="48" height="20" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="132" y="46" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.8">+ D_X, D_Y</text>
        <text x="132" y="53" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.7">adversarial</text>
        <line x1="96" y1="35" x2="108" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="50" y="70" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.55">无配对图像翻译</text>
      </g>
    </svg>
  );
}
