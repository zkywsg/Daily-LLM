import type { MiniArchProps } from "./types";

export function MiniWord2Vec({
  width = 160,
  height = 70,
  ariaLabel = "Word2Vec 架构缩图",
}: MiniArchProps) {
  // Skip-gram:center → embedding → predict context + negative samples
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* center word("king") */}
        <rect x="2" y="30" width="22" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="13" y="40" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">king</text>
        {/* embedding lookup */}
        <rect x="30" y="22" width="24" height="26" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="42" y="32" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">W_in</text>
        <text x="42" y="40" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">300d</text>
        {/* v_center 向量 */}
        <rect x="60" y="28" width="6" height="14" rx="1"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="63" y="50" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.75">v_c</text>
        {/* 正样本:上下文词 */}
        <rect x="80" y="6" width="38" height="10" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="99" y="13" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.85">queen ✓</text>
        {/* 负样本(NEG):3 个随机词 */}
        <rect x="80" y="22" width="38" height="9" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.4" />
        <text x="99" y="28" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.6">apple ✗</text>
        <rect x="80" y="36" width="38" height="9" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.4" />
        <text x="99" y="42" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.6">car ✗</text>
        <rect x="80" y="50" width="38" height="9" rx="2"
          className="illustration__proj illustration__proj--ffn" opacity="0.4" />
        <text x="99" y="56" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.6">tree ✗</text>
        {/* 内积箭头 */}
        <line x1="24" y1="35" x2="30" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="54" y1="35" x2="60" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="66" y1="34" x2="80" y2="11"
          className="illustration__branch illustration__branch--v" />
        <line x1="66" y1="35" x2="80" y2="26"
          className="illustration__branch illustration__branch--v" opacity="0.5" />
        <line x1="66" y1="36" x2="80" y2="40"
          className="illustration__branch illustration__branch--v" opacity="0.5" />
        <line x1="66" y1="37" x2="80" y2="54"
          className="illustration__branch illustration__branch--v" opacity="0.5" />
        {/* sigmoid + loss */}
        <rect x="126" y="22" width="30" height="26" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="141" y="32" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">σ + log</text>
        <text x="141" y="40" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.7">NEG</text>
        <line x1="118" y1="35" x2="126" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="68" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.55">
          king-man+woman=queen · 线性语义结构
        </text>
      </g>
    </svg>
  );
}
