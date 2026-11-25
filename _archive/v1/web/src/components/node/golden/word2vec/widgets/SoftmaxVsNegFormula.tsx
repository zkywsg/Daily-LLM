const W = 700;
const H = 340;

// 上半:softmax 全词表 |V|=50000 个 dot+exp,巨大粉色块
// 下半:NEG 1 个正样本 + 5 个负样本 = 6 个 sigmoid,小绿块

export function SoftmaxVsNegFormula() {
  const SOFT_BAR_W = 580;
  const NEG_BAR_W = (SOFT_BAR_W * 6) / 50000; // 真实比例:6/50000 = 0.012%
  const DISPLAY_NEG_W = Math.max(NEG_BAR_W, 8);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Softmax vs Negative Sampling cost comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        每步计算成本对比 — softmax O(V) vs NEG O(K)
      </text>

      {/* 上 softmax */}
      <text x={20} y={56} fontSize={12} fontWeight={700} fill="#831843">Softmax 原始目标</text>
      <text x={20} y={74} fontSize={11} fill="#6b7280">p(w_O | w_I) = exp(v·v') / Σ_{`{w=1..V}`} exp(v·v'_w)</text>
      <rect x={60} y={86} width={SOFT_BAR_W} height={30} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
      <text x={60 + SOFT_BAR_W / 2} y={106} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">
        |V| ≈ 50 000 次 dot + exp + 求和
      </text>
      <text x={60 + SOFT_BAR_W + 10} y={106} fontSize={11} fontWeight={700} fill="#831843">每步</text>
      <text x={W / 2} y={140} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        60 亿词训练:50000 × 60×10⁹ ≈ 3×10¹⁵ 次 dot — 单机几个月
      </text>

      {/* 分隔线 */}
      <line x1={20} y1={170} x2={W - 20} y2={170} stroke="#e5e7eb" strokeWidth={1} />

      {/* 下 NEG */}
      <text x={20} y={200} fontSize={12} fontWeight={700} fill="#065f46">Negative Sampling 改写</text>
      <text x={20} y={218} fontSize={11} fill="#6b7280">log σ(v·v'_pos) + Σ_{`{k=1..K}`} log σ(-v·v'_neg_k)</text>

      {/* 真实比例条 + 放大显示 */}
      <rect x={60} y={232} width={DISPLAY_NEG_W} height={30} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
      <text x={60 + DISPLAY_NEG_W + 12} y={252} fontSize={11} fontWeight={700} fill="#065f46">
        K+1 = 6 次 sigmoid
      </text>

      {/* 放大展示 6 个 sigmoid 节点 */}
      <text x={20} y={290} fontSize={11} fontWeight={600} fill="#374151">展开:</text>
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const x = 80 + i * 90;
        const isPos = i === 0;
        return (
          <g key={i}>
            <circle cx={x} cy={300} r={16} fill={isPos ? "#dbeafe" : "#fce7f3"} stroke={isPos ? "#3b82f6" : "#ec4899"} strokeWidth={1.5} />
            <text x={x} y={304} textAnchor="middle" fontSize={10} fontWeight={700} fill={isPos ? "#1e40af" : "#831843"}>
              {isPos ? "+1" : "-1"}
            </text>
            <text x={x} y={326} textAnchor="middle" fontSize={9} fill="#6b7280">
              {isPos ? "(fox, quick)" : `(fox, neg${i})`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
