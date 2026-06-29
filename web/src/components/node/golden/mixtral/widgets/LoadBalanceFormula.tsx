const W = 700;
const H = 260;

// Load balance auxiliary loss:
//   L_aux = α · N · Σ f_i · P_i
//   f_i = fraction of tokens routed to expert i
//   P_i = mean router prob for expert i
// 让 viewer 直觉:f 和 P 同时大才罚得多,逼 router 散开

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={5} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>}
    </g>
  );
}

export function LoadBalanceFormula() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Load balance auxiliary loss formula">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Auxiliary Load-Balance Loss (Switch Transformer / Mixtral)
      </text>

      {/* 公式 */}
      <text x={W / 2} y={70} textAnchor="middle" fontSize={20} fontFamily="ui-monospace" fontWeight={700} fill="var(--ink-primary)">
        L_aux = α · N · Σ f_i · P_i
      </text>

      {/* 解释 */}
      <Box x={40} y={120} w={180} h={50} fill="#fce7f3" stroke="#ec4899" label="f_i" sub="路由到 expert i 的 token 比例" />
      <text x={130} y={188} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#831843">
        \"被分配多少\"
      </text>

      <Box x={250} y={120} w={180} h={50} fill="#dbeafe" stroke="#3b82f6" label="P_i" sub="router 给 expert i 的平均概率" />
      <text x={340} y={188} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#1e3a8a">
        \"被想分配多少\"
      </text>

      <Box x={460} y={120} w={200} h={50} fill="#ecfdf5" stroke="#10b981" label="f_i · P_i 同时大才罚" sub="逼 router 不要只盯一个 expert" />

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        α 通常 = 0.01 · 太大压制路由表达力 / 太小没用 · 这一招让 MoE 训练才稳得住
      </text>
    </svg>
  );
}
