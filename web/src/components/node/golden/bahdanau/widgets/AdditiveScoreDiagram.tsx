const W = 700;
const H = 320;

interface Props {
  highlight: "input" | "tanh" | "softmax" | "ctx" | null;
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

export function AdditiveScoreDiagram({ highlight }: Props) {
  const dimInput = highlight && highlight !== "input" ? 0.4 : 1;
  const dimTanh = highlight && highlight !== "tanh" ? 0.4 : 1;
  const dimSoft = highlight && highlight !== "softmax" ? 0.4 : 1;
  const dimCtx = highlight && highlight !== "ctx" ? 0.4 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Additive attention single-step diagram">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Additive Attention 单步 t — e_{`{t,i}`} = vᵀ tanh(W_s · s_{`{t-1}`} + W_h · h_i)
      </text>

      {/* inputs */}
      <g opacity={dimInput}>
        <Box x={20} y={60} w={110} h={36} fill="#fef3c7" stroke="#f59e0b" label="s_{t-1}" sub="decoder state" />
        <Box x={20} y={120} w={110} h={32} fill="#dbeafe" stroke="#3b82f6" label="h_1" />
        <Box x={20} y={158} w={110} h={32} fill="#dbeafe" stroke="#3b82f6" label="h_2" />
        <Box x={20} y={196} w={110} h={32} fill="#dbeafe" stroke="#3b82f6" label="h_3" />
        <text x={75} y={250} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#9ca3af">encoder 输出</text>
      </g>

      {/* MLP combine */}
      <g opacity={dimTanh}>
        <Arrow x1={130} y1={78} x2={170} y2={130} color="#f59e0b" id="t-s1" />
        <Arrow x1={130} y1={136} x2={170} y2={135} color="#3b82f6" id="t-h1" />
        <Box x={170} y={120} w={140} h={32} fill="#fce7f3" stroke="#ec4899" label="tanh(W_s s + W_h h_1)" />
        <Box x={170} y={158} w={140} h={32} fill="#fce7f3" stroke="#ec4899" label="tanh(W_s s + W_h h_2)" />
        <Box x={170} y={196} w={140} h={32} fill="#fce7f3" stroke="#ec4899" label="tanh(W_s s + W_h h_3)" />
        <Arrow x1={130} y1={174} x2={170} y2={174} color="#3b82f6" id="t-h2" />
        <Arrow x1={130} y1={212} x2={170} y2={212} color="#3b82f6" id="t-h3" />

        <Box x={340} y={120} w={70} h={32} fill="#ecfdf5" stroke="#10b981" label="e_{t,1}" />
        <Box x={340} y={158} w={70} h={32} fill="#ecfdf5" stroke="#10b981" label="e_{t,2}" />
        <Box x={340} y={196} w={70} h={32} fill="#ecfdf5" stroke="#10b981" label="e_{t,3}" />
        <Arrow x1={310} y1={136} x2={340} y2={136} color="#ec4899" id="t-e1" />
        <Arrow x1={310} y1={174} x2={340} y2={174} color="#ec4899" id="t-e2" />
        <Arrow x1={310} y1={212} x2={340} y2={212} color="#ec4899" id="t-e3" />
        <text x={375} y={250} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#9ca3af">unnormalized scores</text>
      </g>

      {/* softmax */}
      <g opacity={dimSoft}>
        <Box x={440} y={158} w={80} h={36} fill="#fef3c7" stroke="#f59e0b" label="softmax" />
        <Arrow x1={410} y1={136} x2={440} y2={166} color="#10b981" id="s-1" />
        <Arrow x1={410} y1={174} x2={440} y2={176} color="#10b981" id="s-2" />
        <Arrow x1={410} y1={212} x2={440} y2={186} color="#10b981" id="s-3" />

        <text x={480} y={216} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">α_t (∑=1)</text>
      </g>

      {/* context vector */}
      <g opacity={dimCtx}>
        <Arrow x1={520} y1={176} x2={580} y2={176} color="#f59e0b" id="c-1" />
        <Box x={580} y={158} w={100} h={36} fill="#fce7f3" stroke="#ec4899" label="c_t" sub="= Σ α_{t,i} h_i" />
        <text x={630} y={216} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#9ca3af">送入 decoder GRU</text>
      </g>

      {/* trivia 公式 */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        W_s, W_h, v 三个可学参数 · 这是后来 Transformer scaled dot-product attention 的直接祖先
      </text>
    </svg>
  );
}
