const W = 700;
const H = 280;

interface Props {
  step: number;
}

const STEPS = [
  { label: "原始网络图文对", desc: "(I, T_web) — noisy,alt-text 经常和图不匹配" },
  { label: "Filter 判断", desc: "ITM head 判断 T_web 是否真匹配当前图像" },
  { label: "Captioner 生成", desc: "不匹配的样本,用 LM head 生成新 caption T_syn" },
  { label: "Filter 再检查", desc: "再用 ITM 检查 T_syn 是否真匹配,通过则保留" },
  { label: "扩增数据集", desc: "14M → 130M,用扩增后数据重新训练 BLIP" },
];

export function CapFiltPipelineDiagram({ step }: Props) {
  const boxW = 120;
  const gap = 15;
  const startX = (W - STEPS.length * boxW - (STEPS.length - 1) * gap) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CapFilt 数据清洗流程">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        CapFilt — 用模型自己生成 + 过滤训练数据
      </text>

      {STEPS.map((s, i) => {
        const x = startX + i * (boxW + gap);
        const isActive = i <= step;
        const isCurrent = i === step;
        return (
          <g key={s.label}>
            <rect
              x={x}
              y={60}
              width={boxW}
              height={70}
              fill={isCurrent ? "#fef3c7" : isActive ? "#ecfdf5" : "var(--bg-surface)"}
              stroke={isCurrent ? "#f59e0b" : isActive ? "#10b981" : "var(--border)"}
              strokeWidth={isCurrent ? 2.2 : 1}
              rx={6}
            />
            <text x={x + boxW / 2} y={82} textAnchor="middle" fontSize={9} fontWeight={700} fill={isActive ? "var(--ink-primary)" : "var(--ink-muted)"}>
              {s.label}
            </text>
            {i < STEPS.length - 1 && (
              <path d={`M ${x + boxW + 2} 95 L ${x + boxW + gap - 2} 95`} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arrow-capfilt)" />
            )}
          </g>
        );
      })}

      <text x={W / 2} y={180} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
        {STEPS[step].desc}
      </text>

      <defs>
        <marker id="arrow-capfilt" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
