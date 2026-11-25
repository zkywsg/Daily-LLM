const W = 700;
const H = 240;

// RM 训练目标:对每对偏好 (y_w 胜过 y_l),最大化 log σ(r(x, y_w) - r(x, y_l))。
// Bradley-Terry model:把排序转成 \"pairwise 偏好概率\"。
// 用图框 + 公式 + 注释展开。

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

function Arrow({ x1, y1, x2, y2, label }: { x1: number; y1: number; x2: number; y2: number; label?: string }) {
  const id = `bt-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} markerEnd={`url(#${id})`} />
      {label && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#6b7280">{label}</text>}
    </g>
  );
}

export function BradleyTerryFormula() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Bradley-Terry preference loss">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        RM 训练 — Bradley-Terry pairwise 偏好 loss
      </text>

      {/* y_w + y_l → RM */}
      <Box x={30} y={55} w={120} h={40} fill="#ecfdf5" stroke="#10b981" label="y_w" sub="winner response" />
      <Box x={30} y={115} w={120} h={40} fill="#fef2f2" stroke="#dc2626" label="y_l" sub="loser response" />
      <Arrow x1={150} y1={75} x2={220} y2={95} />
      <Arrow x1={150} y1={135} x2={220} y2={115} />

      <Box x={220} y={85} w={130} h={50} fill="#fce7f3" stroke="#ec4899" label="RM r_φ(x, y)" sub="标量打分器" />
      <Arrow x1={350} y1={110} x2={420} y2={110} label="差值" />

      <Box x={420} y={85} w={120} h={50} fill="#dbeafe" stroke="#3b82f6" label="r(x, y_w) − r(x, y_l)" />
      <Arrow x1={540} y1={110} x2={600} y2={110} />

      <Box x={600} y={85} w={80} h={50} fill="#fef3c7" stroke="#f59e0b" label="−log σ(·)" />

      {/* 公式注脚 */}
      <text x={W / 2} y={185} textAnchor="middle" fontSize={12} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)" fontWeight={700}>
        Loss = − E[ log σ(r_φ(x, y_w) − r_φ(x, y_l)) ]
      </text>
      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        最大化 \"winner 比 loser 分数高\" 的对数概率 · 等价于学一个 utility 函数
      </text>
    </svg>
  );
}
