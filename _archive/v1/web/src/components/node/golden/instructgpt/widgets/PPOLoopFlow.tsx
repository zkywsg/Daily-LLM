const W = 700;
const H = 320;

// PPO 循环:
//   prompt → policy (从 SFT 初始化) → response → RM 打分 → PPO 梯度更新 policy
//                          ↕ KL 约束跟 SFT 距离
// 强调 KL 防止 reward hacking 把 policy 学崩。

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

function Arrow({ x1, y1, x2, y2, label, dashed }: { x1: number; y1: number; x2: number; y2: number; label?: string; dashed?: boolean }) {
  const id = `ppo-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} strokeDasharray={dashed ? "4 3" : "none"} markerEnd={`url(#${id})`} />
      {label && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#6b7280">{label}</text>}
    </g>
  );
}

export function PPOLoopFlow() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PPO RLHF loop with KL constraint">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        PPO + KL — RLHF 第三阶段循环
      </text>

      {/* prompts pool */}
      <Box x={20} y={70} w={110} h={50} fill="#fef3c7" stroke="#f59e0b" label="prompts" sub="31K from API" />
      <Arrow x1={130} y1={95} x2={200} y2={95} />

      {/* policy (active LLM,被更新) */}
      <Box x={200} y={70} w={140} h={50} fill="#fce7f3" stroke="#ec4899" label="π_θ (Policy)" sub="从 SFT 初始化" />
      <Arrow x1={340} y1={95} x2={410} y2={95} label="sample" />

      {/* response */}
      <Box x={410} y={70} w={120} h={50} fill="#ecfdf5" stroke="#10b981" label="response" sub="y ~ π_θ(·|x)" />
      <Arrow x1={530} y1={95} x2={600} y2={95} />

      {/* RM */}
      <Box x={600} y={70} w={80} h={50} fill="#dbeafe" stroke="#3b82f6" label="RM" sub="r_φ(x, y)" />

      {/* RM → reward 反馈到 policy */}
      <Arrow x1={640} y1={120} x2={640} y2={180} />
      <Box x={530} y={180} w={150} h={50} fill="#fce7f3" stroke="#ec4899" label="PPO 梯度更新" sub="∇_θ J(π_θ)" />

      {/* PPO 回 policy */}
      <Arrow x1={530} y1={210} x2={270} y2={210} dashed />
      <Arrow x1={270} y1={210} x2={270} y2={120} dashed />

      {/* SFT 模型(frozen,用于 KL) */}
      <Box x={20} y={180} w={140} h={50} fill="#f3f4f6" stroke="#9ca3af" label="SFT (frozen)" sub="π_SFT" />
      <Arrow x1={90} y1={180} x2={200} y2={140} dashed label="KL 约束" />

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        KL[π_θ || π_SFT] 罚项 — 不让 policy 离 SFT 太远 · 防 reward hacking 把 RM 玩坏
      </text>
    </svg>
  );
}
