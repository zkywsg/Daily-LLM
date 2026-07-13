const W = 700;
const H = 200;

// 预训练浅版 seeding:VGG-11(随机初始化可收敛)-> 迁移卷积权重 -> VGG-13 -> VGG-16 -> VGG-19
const RELAY = [
  { name: "VGG-11", note: "随机初始化直接收敛" },
  { name: "VGG-13", note: "用 VGG-11 权重初始化对应层" },
  { name: "VGG-16", note: "接力初始化,新增层随机初始化" },
  { name: "VGG-19", note: "接力初始化,新增层随机初始化" },
];

export function SeedRelayDiagram() {
  const n = RELAY.length;
  const colW = 130;
  const gap = 30;
  const startX = 50;
  const y = 90;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="VGG 预训练浅版 seeding 接力训练流程"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        预训练浅版 seeding — BN 出现前训深网的接力方案
      </text>

      {RELAY.map((step, i) => {
        const x = startX + i * (colW + gap);
        return (
          <g key={step.name}>
            <rect x={x} y={y - 22} width={colW} height={44} fill={i === 0 ? "#ecfdf5" : "#fce7f3"} stroke={i === 0 ? "#10b981" : "#ec4899"} strokeWidth={1.6} rx={6} />
            <text x={x + colW / 2} y={y - 2} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
              {step.name}
            </text>
            <text x={x + colW / 2} y={y + 14} textAnchor="middle" fontSize={8.5} fill="var(--ink-muted)">
              {step.note}
            </text>
            {i < n - 1 && (
              <line
                x1={x + colW + 4}
                y1={y}
                x2={x + colW + gap - 4}
                y2={y}
                stroke="var(--ink-muted)"
                strokeWidth={1.6}
                markerEnd="url(#arrow-seed)"
              />
            )}
          </g>
        );
      })}

      <defs>
        <marker id="arrow-seed" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--ink-muted)" />
        </marker>
      </defs>

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
        权重迁移只发生在卷积层;每一代新增的层用随机初始化,在已有权重基础上继续训
      </text>
    </svg>
  );
}
