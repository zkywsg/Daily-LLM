import type { MiniArchProps } from "./types";

export function MiniSparselyGatedMoE({
  width = 160,
  height = 70,
  ariaLabel = "Sparsely-Gated MoE 架构缩图",
}: MiniArchProps) {
  // x → gate(top-K) → N 个 expert(只激活几个)→ 加权和
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
        {/* Gate */}
        <rect x="22" y="22" width="22" height="26" rx="2"
          className="illustration__proj illustration__proj--ffn" />
        <text x="33" y="32" textAnchor="middle" fontSize="5" fontWeight="600"
          fill="currentColor" opacity="0.85">Gate</text>
        <text x="33" y="40" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">top-K=4</text>
        {/* 8 个 expert,4 个高亮(激活),4 个灰(未选) */}
        {[
          { y: 6, active: false },
          { y: 14, active: true },
          { y: 22, active: false },
          { y: 30, active: true },
          { y: 38, active: false },
          { y: 46, active: true },
          { y: 54, active: false },
          { y: 62, active: true },
        ].map((e, i) => (
          e.active
            ? <rect key={i} x="52" y={e.y} width="40" height="6" rx="1"
                className="illustration__featuremap illustration__featuremap--ctx" />
            : <rect key={i} x="52" y={e.y} width="40" height="6" rx="1"
                className="illustration__proj illustration__proj--ffn" opacity="0.25" />
        ))}
        <text x="72" y="3" textAnchor="middle" fontSize="4.5" fontWeight="600"
          fill="currentColor" opacity="0.7">N experts</text>
        {/* Gate 到 expert 的箭头(只画 active) */}
        <line x1="44" y1="35" x2="52" y2="17"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="35" x2="52" y2="33"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="35" x2="52" y2="49"
          className="illustration__branch illustration__branch--q" />
        <line x1="44" y1="35" x2="52" y2="65"
          className="illustration__branch illustration__branch--q" />
        {/* Σ */}
        <rect x="100" y="22" width="24" height="26" rx="3"
          className="illustration__proj illustration__proj--v" />
        <text x="112" y="34" textAnchor="middle" fontSize="9" fontWeight="600"
          fill="currentColor" opacity="0.85">Σ</text>
        <text x="112" y="43" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.7">gᵢ·Eᵢ</text>
        {/* expert 到 Σ(只画 active) */}
        <line x1="92" y1="17" x2="100" y2="29"
          className="illustration__branch illustration__branch--v" />
        <line x1="92" y1="33" x2="100" y2="33"
          className="illustration__branch illustration__branch--v" />
        <line x1="92" y1="49" x2="100" y2="37"
          className="illustration__branch illustration__branch--v" />
        <line x1="92" y1="65" x2="100" y2="42"
          className="illustration__branch illustration__branch--v" />
        {/* y */}
        <rect x="132" y="30" width="14" height="10" rx="2"
          className="illustration__proj illustration__proj--v" />
        <text x="139" y="38" textAnchor="middle" fontSize="6" fontWeight="600"
          fill="currentColor" opacity="0.85">y</text>
        <line x1="124" y1="35" x2="132" y2="35"
          className="illustration__branch illustration__branch--v" />
        {/* 底部 */}
        <text x="80" y="69" textAnchor="middle" fontSize="4.5"
          fill="currentColor" opacity="0.5">
          稀疏激活 · aux loss 防 expert 塌缩
        </text>
      </g>
    </svg>
  );
}
