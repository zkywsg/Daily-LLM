interface Props {
  d: number;
  k: number;
  r: number;
}

const W = 700;
const H = 280;

// 把 W = W₀ + (α/r)·B·A 画成"大矩形 + 小矩形相乘 = 大矩形 ΔW"。
// 矩形尺寸用 sqrt(d) / sqrt(k) 等做对数缩放,viewer 直接看到 r 变小时
// B 和 A 一起塌成"细高 × 矮宽 → 小到几乎没有"。
// 这是 LoRA 论文 figure 1 的卡通版,关键是让 viewer 摸到"参数预算"在哪。

const PIXELS_PER_DIM = 0.5; // 1 dim ≈ 0.5 px(基础值,够直观)

export function WeightDecompositionSVG({ d, k, r }: Props) {
  // 画布坐标系
  const wW0 = Math.max(60, k * PIXELS_PER_DIM);
  const hW0 = Math.max(60, d * PIXELS_PER_DIM);
  const wA = Math.max(60, k * PIXELS_PER_DIM);
  const hA = Math.max(12, r * 4);
  const wB = Math.max(12, r * 4);
  const hB = Math.max(60, d * PIXELS_PER_DIM);

  const W0X = 30;
  const W0Y = (H - hW0) / 2;
  const eqX = W0X + wW0 + 40;
  const eqY = H / 2;
  const BX = eqX + 30;
  const BY = (H - hB) / 2;
  const AX = BX + wB + 10;
  const AY = (H - hA) / 2;
  const DWX = AX + wA + 60;
  const DWY = (H - hW0) / 2;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Weight decomposition W = W0 + BA, rank ${r}`}
    >
      {/* W₀(冻结,灰色) */}
      <rect
        x={W0X}
        y={W0Y}
        width={wW0}
        height={hW0}
        fill="#e5e7eb"
        stroke="#6b7280"
        strokeWidth={1.5}
        rx={3}
      />
      <text x={W0X + wW0 / 2} y={W0Y + hW0 / 2 - 4} textAnchor="middle" fontSize={13} fontWeight={600} fill="#1f2937">
        W₀
      </text>
      <text x={W0X + wW0 / 2} y={W0Y + hW0 / 2 + 12} textAnchor="middle" fontSize={10} fill="#6b7280">
        {d} × {k} · frozen
      </text>

      {/* + (alpha/r) · 写在 W₀ 和 BA 中间 */}
      <text x={eqX} y={eqY + 4} textAnchor="middle" fontSize={20} fontWeight={600} fill="var(--ink-primary)">
        +
      </text>
      <text x={eqX} y={eqY + 24} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        α/r
      </text>

      {/* B(细高,trainable 蓝) */}
      <rect
        x={BX}
        y={BY}
        width={wB}
        height={hB}
        fill="#dbeafe"
        stroke="#3b82f6"
        strokeWidth={1.5}
        rx={3}
      />
      <text x={BX + wB / 2} y={BY - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1d4ed8">
        B
      </text>
      <text x={BX + wB / 2} y={BY + hB + 14} textAnchor="middle" fontSize={9} fill="#3b82f6">
        {d}×{r}
      </text>

      {/* · 写在 B 和 A 之间 */}
      <text x={BX + wB + 3} y={H / 2 + 4} fontSize={16} fontWeight={600} fill="var(--ink-primary)">
        ·
      </text>

      {/* A(矮宽,trainable 蓝) */}
      <rect
        x={AX}
        y={AY}
        width={wA}
        height={hA}
        fill="#dbeafe"
        stroke="#3b82f6"
        strokeWidth={1.5}
        rx={3}
      />
      <text x={AX + wA / 2} y={AY - 6} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1d4ed8">
        A
      </text>
      <text x={AX + wA / 2} y={AY + hA + 14} textAnchor="middle" fontSize={9} fill="#3b82f6">
        {r}×{k}
      </text>

      {/* = ΔW(粉色 compute,推理时合并回 W₀) */}
      <text x={AX + wA + 28} y={H / 2 + 4} textAnchor="middle" fontSize={16} fontWeight={600} fill="var(--ink-primary)">
        =
      </text>
      <rect
        x={DWX}
        y={DWY}
        width={wW0}
        height={hW0}
        fill="#fce7f3"
        stroke="#ec4899"
        strokeWidth={1.5}
        rx={3}
      />
      <text x={DWX + wW0 / 2} y={DWY + hW0 / 2 - 4} textAnchor="middle" fontSize={13} fontWeight={600} fill="#1f2937">
        ΔW
      </text>
      <text x={DWX + wW0 / 2} y={DWY + hW0 / 2 + 12} textAnchor="middle" fontSize={10} fill="#831843">
        rank ≤ {r}
      </text>

      {/* 顶部说明 */}
      <text x={W / 2} y={18} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        W = W₀ + (α/r) · B · A
      </text>
    </svg>
  );
}
