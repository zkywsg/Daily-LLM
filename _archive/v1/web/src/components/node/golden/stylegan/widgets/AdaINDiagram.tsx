const W = 700;
const H = 340;

interface Props {
  highlight: "instnorm" | "affine" | "modulate" | null;
}

function Box({ x, y, w, h, fill, stroke, label, sub, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function AdaINDiagram({ highlight }: Props) {
  const dimIN = highlight && highlight !== "instnorm" ? 0.4 : 1;
  const dimAff = highlight && highlight !== "affine" ? 0.4 : 1;
  const dimMod = highlight && highlight !== "modulate" ? 0.4 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="AdaIN diagram">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        AdaIN — 用 style 控制每层 feature 的统计量
      </text>

      {/* 输入 feature x */}
      <Box x={30} y={70} w={140} h={40} fill="#dbeafe" stroke="#3b82f6" label="x (feature map)" sub="B × C × H × W" />

      {/* Instance Norm step */}
      <g opacity={dimIN}>
        <Arrow x1={170} y1={90} x2={210} y2={90} color="#3b82f6" id="in-a" />
        <Box x={210} y={70} w={180} h={40} fill="#ecfdf5" stroke="#10b981"
             label="Instance Normalize" sub="(x - μ) / σ · 擦掉统计" />
      </g>

      {/* w → affine → (scale, bias) */}
      <g opacity={dimAff}>
        <Box x={30} y={170} w={140} h={40} fill="#fef3c7" stroke="#f59e0b" label="w (style)" sub="512 维" />
        <Arrow x1={170} y1={190} x2={210} y2={190} color="#f59e0b" id="aff-a" />
        <Box x={210} y={170} w={180} h={40} fill="#fce7f3" stroke="#ec4899"
             label="affine (learned linear)" sub="w → (scale, bias)" />
        <Arrow x1={390} y1={190} x2={430} y2={190} color="#ec4899" id="aff-b" />
        <Box x={430} y={162} w={100} h={22} fill="#fce7f3" stroke="#ec4899" label="scale y_s" />
        <Box x={430} y={198} w={100} h={22} fill="#fce7f3" stroke="#ec4899" label="bias y_b" />
      </g>

      {/* Modulate */}
      <g opacity={dimMod}>
        <Arrow x1={390} y1={90} x2={480} y2={90} color="#10b981" id="mod-x" />
        <Arrow x1={480} y1={162} x2={480} y2={110} color="#ec4899" id="mod-s" />
        <Arrow x1={530} y1={198} x2={530} y2={110} color="#ec4899" id="mod-b" />

        <Box x={470} y={70} w={140} h={40} fill="#fce7f3" stroke="#ec4899"
             label="y_s · x + y_b" sub="重新调制" />

        <Arrow x1={610} y1={90} x2={640} y2={90} color="#ec4899" id="mod-out" />
        <text x={648} y={94} fontSize={10} fill="#831843">→ 下一层</text>
      </g>

      {/* 公式 */}
      <rect x={80} y={252} width={W - 160} height={40} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} opacity={0.6} />
      <text x={W / 2} y={276} textAnchor="middle" fontSize={12} fontFamily="ui-monospace, monospace" fill="#1f2937">
        AdaIN(x, w) = y_s · (x − μ(x)) / σ(x) + y_b
      </text>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        style 改变 feature 统计量 → 改变外观(颜色/纹理) · 不改变空间结构(位置/形状)
      </text>
    </svg>
  );
}
