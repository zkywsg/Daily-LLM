import type { MiniArchProps } from "./types";

export function MiniELMo({
  width = 160,
  height = 70,
  ariaLabel = "ELMo 架构缩图",
}: MiniArchProps) {
  // char-CNN → 两层 biLSTM(→ ← 双向)→ 加权和
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* 输入:char-level */}
        <rect x="2" y="30" width="20" height="14" rx="2"
          className="illustration__layer illustration__layer--input" />
        <text x="12" y="38" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">chars</text>
        <text x="12" y="43" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">b-a-n-k</text>
        {/* char-CNN */}
        <rect x="26" y="22" width="18" height="26" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="35" y="31" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">char</text>
        <text x="35" y="37" textAnchor="middle" fontSize="4"
          fill="currentColor" opacity="0.75">CNN</text>
        <text x="35" y="44" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.65">h₀</text>
        {/* biLSTM Layer 1(双向) */}
        <rect x="50" y="14" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="61" y="22" textAnchor="middle" fontSize="4" fontWeight="600"
          fill="currentColor" opacity="0.85">LSTM₁→</text>
        <text x="61" y="26" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">h₁</text>
        <rect x="50" y="42" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="61" y="50" textAnchor="middle" fontSize="4" fontWeight="600"
          fill="currentColor" opacity="0.85">←LSTM₁</text>
        {/* biLSTM Layer 2 */}
        <rect x="78" y="14" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="89" y="22" textAnchor="middle" fontSize="4" fontWeight="600"
          fill="currentColor" opacity="0.85">LSTM₂→</text>
        <text x="89" y="26" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">h₂</text>
        <rect x="78" y="42" width="22" height="14" rx="2"
          className="illustration__featuremap illustration__featuremap--ctx" />
        <text x="89" y="50" textAnchor="middle" fontSize="4" fontWeight="600"
          fill="currentColor" opacity="0.85">←LSTM₂</text>
        {/* 加权和(task-specific) */}
        <rect x="108" y="24" width="24" height="22" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="120" y="33" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">Σ sⱼ·hⱼ</text>
        <text x="120" y="40" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">task-weight</text>
        {/* ELMo 输出向量 */}
        <rect x="138" y="30" width="18" height="14" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="147" y="38" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.85">ELMo</text>
        <text x="147" y="43" textAnchor="middle" fontSize="3.5"
          fill="currentColor" opacity="0.7">ctx</text>
        {/* 箭头 */}
        <line x1="22" y1="37" x2="26" y2="37"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="30" x2="50" y2="21"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="40" x2="50" y2="49"
          className="illustration__branch illustration__branch--q" />
        <line x1="72" y1="21" x2="78" y2="21"
          className="illustration__branch illustration__branch--q" />
        <line x1="72" y1="49" x2="78" y2="49"
          className="illustration__branch illustration__branch--v" />
        {/* h₀/h₁/h₂ → Σ(三层加权) */}
        <line x1="44" y1="35" x2="108" y2="32"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--q" opacity="0.55" />
        <line x1="72" y1="21" x2="108" y2="32"
          strokeDasharray="2 2"
          className="illustration__branch illustration__branch--q" opacity="0.55" />
        <line x1="100" y1="21" x2="108" y2="34"
          className="illustration__branch illustration__branch--v" />
        <line x1="100" y1="49" x2="108" y2="38"
          className="illustration__branch illustration__branch--v" />
        <line x1="132" y1="35" x2="138" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="69" textAnchor="middle" fontSize="3.8"
          fill="currentColor" opacity="0.55">
          双向 biLM · contextualized · "river bank" ≠ "money bank"
        </text>
      </g>
    </svg>
  );
}
