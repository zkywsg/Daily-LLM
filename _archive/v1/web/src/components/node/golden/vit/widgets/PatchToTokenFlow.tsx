const W = 700;
const H = 240;

// patch (16×16×3) → flatten 成 768 维 → linear projection → d_model 维 token。
// 流程图风格,带每步形状标注。

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, label }: { x1: number; y1: number; x2: number; y2: number; label?: string }) {
  const id = `arr-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} markerEnd={`url(#${id})`} />
      {label && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#6b7280">
          {label}
        </text>
      )}
    </g>
  );
}

export function PatchToTokenFlow() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Patch to token embedding flow">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        单个 patch → token 向量:flatten → linear projection
      </text>

      {/* patch 立方块 */}
      <Box x={30} y={70} w={110} h={70} fill="#fef3c7" stroke="#f59e0b" label="patch" sub="16 × 16 × 3" />
      <Arrow x1={140} y1={105} x2={210} y2={105} label="flatten" />

      {/* flat vector */}
      <Box x={210} y={70} w={130} h={70} fill="#fce7f3" stroke="#ec4899" label="flat vector" sub="ℝ⁷⁶⁸ (= 16·16·3)" />
      <Arrow x1={340} y1={105} x2={410} y2={105} label="W_e linear" />

      {/* token */}
      <Box x={410} y={70} w={130} h={70} fill="#dbeafe" stroke="#3b82f6" label="patch token" sub="ℝ^d_model" />
      <Arrow x1={540} y1={105} x2={610} y2={105} />

      {/* + pos emb */}
      <Box x={610} y={70} w={70} h={70} fill="#ecfdf5" stroke="#10b981" label="+ pos" sub="emb" />

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        N 个 patch token + 1 个 [CLS] token + N+1 个可学位置嵌入 → Transformer encoder
      </text>
    </svg>
  );
}
