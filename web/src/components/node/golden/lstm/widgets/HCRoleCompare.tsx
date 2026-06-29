const W = 700;
const H = 280;

// 对比卡:h vs C 各自承担什么角色,谁给输出层,谁负责长程。

function Card({
  x, y, w, h, fill, stroke, title, bullets,
}: {
  x: number; y: number; w: number; h: number;
  fill: string; stroke: string; title: string; bullets: string[];
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + 22} textAnchor="middle" fontSize={13} fontWeight={700} fill="#1f2937">{title}</text>
      {bullets.map((b, i) => (
        <text key={i} x={x + 14} y={y + 50 + i * 22} fontSize={11} fill="#374151">
          • {b}
        </text>
      ))}
    </g>
  );
}

export function HCRoleCompare() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Hidden state vs Cell state role comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        h vs C 分工 — 一个对外发声 / 一个对内长记
      </text>

      <Card
        x={20}
        y={40}
        w={320}
        h={210}
        fill="#ecfdf5"
        stroke="#10b981"
        title="h (Hidden State)"
        bullets={[
          "= o · tanh(C)  → 每步都 reshape",
          "给输出层 (vocab logits / 下游任务)",
          "短期可塑性强,一步一变",
          "对外的 \"表达通道\"",
          "维度 = hidden_size",
          "类比:你说出口的话",
        ]}
      />
      <Card
        x={360}
        y={40}
        w={320}
        h={210}
        fill="#fce7f3"
        stroke="#ec4899"
        title="C (Cell State)"
        bullets={[
          "= f·C_{ₜ₋₁} + i·g  → 累积式更新",
          "不直接输出,只在内部 timestep 传递",
          "长程稳定,梯度走 highway 不消失",
          "对内的 \"长程记忆\"",
          "维度 = hidden_size (跟 h 一致)",
          "类比:你心里默默记的事",
        ]}
      />

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        两条通道独立各管一面 — output gate 决定哪些 C 露面给外界
      </text>
    </svg>
  );
}
