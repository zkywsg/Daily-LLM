import type { MiniArchProps } from "./types";

export function MiniRAG({
  width = 160,
  height = 70,
  ariaLabel = "RAG 架构缩图",
}: MiniArchProps) {
  // Q → retriever → top-K docs → generator → A
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* Q */}
        <rect x="2" y="28" width="14" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="9" y="38" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.8">Q</text>
        {/* Retriever(DPR 双塔) */}
        <rect x="22" y="22" width="22" height="26" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="33" y="32" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">DPR</text>
        <text x="33" y="40" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">dense</text>
        {/* KB 知识库柱状(代表向量索引) */}
        <rect x="50" y="6" width="22" height="58" rx="2"
          fill="none"
          className="illustration__block" />
        <text x="61" y="13" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">KB</text>
        <rect x="53" y="16" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.7" />
        <rect x="53" y="22" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.9" />
        <rect x="53" y="28" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.5" />
        <rect x="53" y="34" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.9" />
        <rect x="53" y="40" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.6" />
        <rect x="53" y="46" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.8" />
        <rect x="53" y="52" width="16" height="4" rx="0.5"
          className="illustration__featuremap illustration__featuremap--ctx" opacity="0.45" />
        <text x="61" y="62" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.6">top-K</text>
        {/* top-K docs 选中的几条 */}
        <rect x="78" y="22" width="20" height="26" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="88" y="32" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.8">docs</text>
        <text x="88" y="40" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">k=5</text>
        {/* Generator(BART) */}
        <rect x="104" y="14" width="34" height="42" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="121" y="30" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">BART</text>
        <text x="121" y="40" textAnchor="middle" fontSize="5"
          fill="currentColor" opacity="0.7">seq2seq</text>
        {/* A */}
        <rect x="144" y="28" width="14" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="151" y="38" textAnchor="middle" fontSize="6.5" fontWeight="600"
          fill="currentColor" opacity="0.85">A</text>
        {/* 箭头 */}
        <line x1="16" y1="35" x2="22" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="35" x2="50" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="72" y1="35" x2="78" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="98" y1="35" x2="104" y2="35"
          className="illustration__branch illustration__branch--q" />
        <line x1="138" y1="35" x2="144" y2="35"
          className="illustration__branch illustration__branch--v" />
      </g>
    </svg>
  );
}
