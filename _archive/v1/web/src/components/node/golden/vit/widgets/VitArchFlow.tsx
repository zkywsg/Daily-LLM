const W = 700;
const H = 320;

// ViT 全 pipeline:
//   patches + [CLS] + pos emb → N×encoder block → CLS 输出 → MLP head → class logits

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={5} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  const id = `va-arr-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function VitArchFlow() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ViT architecture pipeline">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        ViT 整体架构 — CLS token 是 \"句子级\" 分类信号
      </text>

      {/* Token 序列输入 */}
      <g transform="translate(30, 70)">
        <text x={0} y={-8} fontSize={10} fill="var(--ink-muted)">输入 tokens</text>
        {/* CLS */}
        <rect x={0} y={0} width={36} height={28} rx={3} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
        <text x={18} y={18} textAnchor="middle" fontSize={10} fontWeight={700} fill="#831843">CLS</text>
        {/* patch tokens */}
        {Array.from({ length: 6 }, (_, i) => (
          <g key={i}>
            <rect x={42 + i * 32} y={0} width={28} height={28} rx={3} fill="#fef3c7" stroke="#f59e0b" />
            <text x={56 + i * 32} y={18} textAnchor="middle" fontSize={10} fill="#92400e">t{i + 1}</text>
          </g>
        ))}
      </g>

      {/* + pos emb */}
      <g transform="translate(30, 116)">
        <rect x={0} y={0} width={244} height={20} rx={3} fill="#ecfdf5" stroke="#10b981" />
        <text x={122} y={14} textAnchor="middle" fontSize={10} fontWeight={600} fill="#065f46">+ 可学位置嵌入 (N+1, d_model)</text>
      </g>

      {/* 箭头 ↓ */}
      <Arrow x1={152} y1={142} x2={152} y2={170} />

      {/* Encoder block × N */}
      <Box x={30} y={170} w={244} h={50} fill="#dbeafe" stroke="#3b82f6" label="N × Transformer Encoder Block" sub="LayerNorm → MSA → MLP → 残差" />

      {/* 输出 token 序列 */}
      <Arrow x1={152} y1={220} x2={152} y2={250} />
      <g transform="translate(30, 250)">
        <rect x={0} y={0} width={36} height={28} rx={3} fill="#fce7f3" stroke="#ec4899" strokeWidth={2.5} />
        <text x={18} y={18} textAnchor="middle" fontSize={10} fontWeight={700} fill="#831843">CLS</text>
        <text x={18} y={48} textAnchor="middle" fontSize={8} fontStyle="italic" fill="#831843">取这个</text>
        {Array.from({ length: 6 }, (_, i) => (
          <g key={i}>
            <rect x={42 + i * 32} y={0} width={28} height={28} rx={3} fill="#fef3c7" stroke="#f59e0b" opacity={0.4} />
            <text x={56 + i * 32} y={18} textAnchor="middle" fontSize={10} fill="#92400e" opacity={0.4}>t{i + 1}'</text>
          </g>
        ))}
      </g>

      {/* CLS → MLP head → logits */}
      <Arrow x1={310} y1={282} x2={380} y2={282} />
      <Box x={380} y={262} w={130} h={40} fill="#fce7f3" stroke="#ec4899" label="MLP Head" sub="linear classifier" />
      <Arrow x1={510} y1={282} x2={570} y2={282} />
      <Box x={570} y={262} w={100} h={40} fill="#ecfdf5" stroke="#10b981" label="logits" sub="1000 类" />

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        CLS 在编码过程中聚合了所有 patch 的全局信息 — 取它的输出做分类
      </text>
    </svg>
  );
}
