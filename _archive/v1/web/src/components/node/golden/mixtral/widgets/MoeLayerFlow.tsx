const W = 700;
const H = 320;

// 一个 MoE 层的数据流:
//   token x → router (linear + softmax) → top-2 expert weights
//          → 选中的 2 个 expert 各跑一次 FFN
//          → 按 weight 加权求和 → y

function Box({
  x, y, w, h, fill, stroke, label, sub, faded,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string;
  label: string; sub?: string; faded?: boolean;
}) {
  return (
    <g opacity={faded ? 0.35 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, label, faded, color }: { x1: number; y1: number; x2: number; y2: number; label?: string; faded?: boolean; color?: string }) {
  const c = color ?? "#9ca3af";
  const id = `moe-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g opacity={faded ? 0.35 : 1}>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={c} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={1.6} markerEnd={`url(#${id})`} />
      {label && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill={c}>{label}</text>}
    </g>
  );
}

export function MoeLayerFlow() {
  // 8 个 expert,选中 #1 和 #4 当 demo
  const selected = new Set([1, 4]);
  const expertYStart = 40;
  const expertGap = 28;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="MoE layer data flow">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Sparse MoE Layer:router → top-2 expert → 加权求和
      </text>

      {/* 输入 token */}
      <Box x={20} y={140} w={100} h={50} fill="#fef3c7" stroke="#f59e0b" label="token x" sub="hidden state" />
      <Arrow x1={120} y1={165} x2={190} y2={165} label="x" />

      {/* router */}
      <Box x={190} y={140} w={130} h={50} fill="#fce7f3" stroke="#ec4899" label="Router" sub="W_r·x → softmax" />

      {/* router → 每个 expert(8 条线;选中的高亮,未选的虚弱) */}
      {Array.from({ length: 8 }, (_, e) => (
        <Arrow
          key={`r-e-${e}`}
          x1={320}
          y1={165}
          x2={400}
          y2={expertYStart + expertGap * e + 12}
          faded={!selected.has(e)}
          color={selected.has(e) ? "#ec4899" : "#d1d5db"}
        />
      ))}

      {/* 8 个 expert */}
      {Array.from({ length: 8 }, (_, e) => {
        const isSelected = selected.has(e);
        return (
          <Box
            key={`exp-${e}`}
            x={400}
            y={expertYStart + expertGap * e}
            w={120}
            h={22}
            fill={isSelected ? "#dbeafe" : "#f3f4f6"}
            stroke={isSelected ? "#3b82f6" : "#d1d5db"}
            label={`expert ${e}`}
            faded={!isSelected}
          />
        );
      })}

      {/* 选中 expert → 加权求和 */}
      {[1, 4].map((e, idx) => {
        const w = idx === 0 ? 0.62 : 0.38;
        return (
          <Arrow
            key={`merge-${e}`}
            x1={520}
            y1={expertYStart + expertGap * e + 12}
            x2={600}
            y2={165}
            color="#ec4899"
            label={w.toFixed(2)}
          />
        );
      })}

      {/* 输出 y */}
      <Box x={600} y={140} w={80} h={50} fill="#ecfdf5" stroke="#10b981" label="y" sub="= Σ w_i·E_i(x)" />

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        Mixtral 8x7B:8 个 expert · 每 token 选 top-2 (k=2) · 算力 ≈ 2/8 dense MoE
      </text>
    </svg>
  );
}
