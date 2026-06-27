interface Props {
  seqLen: number;
}

const W = 700;
const H = 220;

// 计算 attention FLOPs 复杂度对比:
//   dense: O(n²)
//   sparse: O(n√n)
// 在 n=2048 / n=8192 这种长 context 下差好几十倍 —— GPT-3 用 sparse 是为了能塞长 context。
export function ComputeBudgetBar({ seqLen }: Props) {
  const dense = seqLen * seqLen;
  const sparse = seqLen * Math.sqrt(seqLen);
  const ratio = dense / sparse;
  const max = dense;
  const px = (v: number) => (v / max) * (W - 220);

  const fmtFlops = (v: number) => {
    if (v >= 1e9) return `${(v / 1e9).toFixed(1)}G`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return v.toFixed(0);
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Attention compute: dense vs sparse">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Attention 计算复杂度 · seq_len = {seqLen}
      </text>

      {/* Dense */}
      <text x={20} y={62} fontSize={11} fontWeight={600} fill="var(--ink-primary)">Dense O(n²)</text>
      <rect x={120} y={50} width={Math.max(2, px(dense))} height={24} rx={3} fill="#9ca3af" opacity={0.8} />
      <text x={130 + px(dense)} y={67} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
        ≈ {fmtFlops(dense)} ops
      </text>

      {/* Sparse */}
      <text x={20} y={112} fontSize={11} fontWeight={600} fill="var(--ink-primary)">Sparse O(n√n)</text>
      <rect x={120} y={100} width={Math.max(2, px(sparse))} height={24} rx={3} fill="#ec4899" opacity={0.85} />
      <text x={130 + px(sparse)} y={117} fontSize={11} fontWeight={600} fill="#831843">
        ≈ {fmtFlops(sparse)} ops · 省 {ratio.toFixed(1)}×
      </text>

      {/* 说明 */}
      <text x={W / 2} y={170} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-secondary)">
        n=2048 时 dense 4M ops / sparse 92K ops · 差 45×
      </text>
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        GPT-3 用 50% 的 layer 走 sparse —— 在保留质量的前提下省一半 attention 算力
      </text>
    </svg>
  );
}
