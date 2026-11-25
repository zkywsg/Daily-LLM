const W = 700;
const H = 360;

interface Props {
  highlight: "actor" | "ref" | "logratio" | "loss" | null;
}

function Box({ x, y, w, h, fill, stroke, label, sub, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, id, dashed }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string; dashed?: boolean }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} strokeDasharray={dashed ? "4 3" : undefined} />
    </g>
  );
}

export function DpoLossDataflow({ highlight }: Props) {
  const dimActor = highlight && highlight !== "actor" ? 0.4 : 1;
  const dimRef = highlight && highlight !== "ref" ? 0.4 : 1;
  const dimLR = highlight && highlight !== "logratio" ? 0.4 : 1;
  const dimLoss = highlight && highlight !== "loss" ? 0.4 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DPO loss data flow">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        DPO Loss 数据流 — 一条 (prompt, y_w, y_l) 怎么变成 loss
      </text>

      {/* 输入 */}
      <Box x={20}  y={70} w={130} h={40} fill="#fef3c7" stroke="#f59e0b" label="prompt x" />
      <Box x={20}  y={130} w={130} h={40} fill="#ecfdf5" stroke="#10b981" label="y_w (chosen)" />
      <Box x={20}  y={190} w={130} h={40} fill="#fce7f3" stroke="#ec4899" label="y_l (rejected)" />

      {/* actor (训练中) */}
      <g opacity={dimActor}>
        <Box x={200} y={70} w={130} h={70} fill="#dbeafe" stroke="#3b82f6" label="actor π_θ" sub="trainable" />
        <Arrow x1={150} y1={90} x2={200} y2={90} color="#3b82f6" id="a1" />
        <Arrow x1={150} y1={150} x2={200} y2={130} color="#3b82f6" id="a2" />
        <Arrow x1={150} y1={210} x2={200} y2={140} color="#3b82f6" id="a3" />

        <Box x={360} y={70} w={140} h={30} fill="#ecfdf5" stroke="#10b981" label="log π_θ(y_w | x)" />
        <Box x={360} y={110} w={140} h={30} fill="#fce7f3" stroke="#ec4899" label="log π_θ(y_l | x)" />
        <Arrow x1={330} y1={95} x2={360} y2={85} color="#3b82f6" id="a4" />
        <Arrow x1={330} y1={130} x2={360} y2={125} color="#3b82f6" id="a5" />
      </g>

      {/* ref (冻结) */}
      <g opacity={dimRef}>
        <Box x={200} y={170} w={130} h={70} fill="#f3f4f6" stroke="#9ca3af" label="π_ref" sub="frozen (SFT init)" />
        <Arrow x1={150} y1={150} x2={200} y2={180} color="#9ca3af" id="r1" dashed />
        <Arrow x1={150} y1={210} x2={200} y2={220} color="#9ca3af" id="r2" dashed />

        <Box x={360} y={170} w={140} h={30} fill="#f3f4f6" stroke="#9ca3af" label="log π_ref(y_w | x)" />
        <Box x={360} y={210} w={140} h={30} fill="#f3f4f6" stroke="#9ca3af" label="log π_ref(y_l | x)" />
        <Arrow x1={330} y1={190} x2={360} y2={185} color="#9ca3af" id="r3" />
        <Arrow x1={330} y1={210} x2={360} y2={225} color="#9ca3af" id="r4" />
      </g>

      {/* log-ratio */}
      <g opacity={dimLR}>
        <Box x={530} y={90} w={130} h={50} fill="#fef3c7" stroke="#f59e0b" label="log(π_θ/π_ref)" sub="for y_w / y_l" />
        <Arrow x1={500} y1={85} x2={530} y2={105} color="#f59e0b" id="lr1" />
        <Arrow x1={500} y1={125} x2={530} y2={120} color="#f59e0b" id="lr2" />
        <Arrow x1={500} y1={185} x2={530} y2={125} color="#f59e0b" id="lr3" />
        <Arrow x1={500} y1={225} x2={530} y2={130} color="#f59e0b" id="lr4" />
      </g>

      {/* loss */}
      <g opacity={dimLoss}>
        <Box x={530} y={170} w={130} h={50} fill="#fce7f3" stroke="#ec4899" label="−log σ(β · margin)" sub="cross-entropy form" />
        <Arrow x1={595} y1={140} x2={595} y2={170} color="#ec4899" id="lo1" />
      </g>

      {/* 公式 */}
      <rect x={50} y={280} width={W - 100} height={50} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} opacity={0.65} />
      <text x={W / 2} y={302} textAnchor="middle" fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">
        L_DPO = − log σ(β · [ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ])
      </text>
      <text x={W / 2} y={320} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#92400e">
        关键奇迹:Z(x) 在 log-ratio 相减时消掉,不需要算难算的归一化
      </text>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        没有 RM · 没有采样 · 没有 PPO clip · 4 个 log-prob → 1 个标量 loss
      </text>
    </svg>
  );
}
