interface Props {
  activeLayer: number; // 1-indexed, which layer is currently "highlighted"
}

const NUM_LAYERS = 8; // 缩略展示(真实 BERT-large 是 24 层)

export function ParameterSharingDiagram({ activeLayer }: Props) {
  const W = 700;
  const H = 300;
  const boxW = 56;
  const boxH = 36;
  const gap = 8;
  const startX = 70;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT 每层独立参数 vs ALBERT 跨层共享参数对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        BERT:24 组独立权重 vs ALBERT:1 组权重重复用 24 次
      </text>

      {/* BERT row */}
      <text x={startX - 20} y={80} textAnchor="end" fontSize={11} fontWeight={700} fill="#3b82f6">BERT</text>
      {Array.from({ length: NUM_LAYERS }, (_, i) => {
        const x = startX + i * (boxW + gap);
        const isActive = i + 1 === activeLayer;
        return (
          <g key={`bert-${i}`}>
            <rect
              x={x}
              y={60}
              width={boxW}
              height={boxH}
              rx={4}
              fill={isActive ? "#dbeafe" : "#f3f4f6"}
              stroke={isActive ? "#3b82f6" : "#9ca3af"}
              strokeWidth={isActive ? 2.4 : 1.2}
            />
            <text x={x + boxW / 2} y={60 + boxH / 2 + 4} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
              W{i + 1}
            </text>
            <text x={x + boxW / 2} y={112} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
              层{i + 1}
            </text>
          </g>
        );
      })}
      <text x={startX + NUM_LAYERS * (boxW + gap) + 6} y={80} fontSize={9} fill="var(--ink-muted)">
        ···(共 24 层,24 组不同权重)
      </text>

      {/* ALBERT row */}
      <text x={startX - 20} y={190} textAnchor="end" fontSize={11} fontWeight={700} fill="#10b981">ALBERT</text>
      {Array.from({ length: NUM_LAYERS }, (_, i) => {
        const x = startX + i * (boxW + gap);
        const isActive = i + 1 === activeLayer;
        return (
          <g key={`albert-${i}`}>
            <rect
              x={x}
              y={170}
              width={boxW}
              height={boxH}
              rx={4}
              fill={isActive ? "#ecfdf5" : "#f3f4f6"}
              stroke={isActive ? "#10b981" : "#9ca3af"}
              strokeWidth={isActive ? 2.4 : 1.2}
              strokeDasharray={isActive ? undefined : "3 2"}
            />
            <text x={x + boxW / 2} y={170 + boxH / 2 + 4} textAnchor="middle" fontSize={9} fontWeight={isActive ? 700 : 400} fill="var(--ink-primary)">
              W
            </text>
            <text x={x + boxW / 2} y={222} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
              层{i + 1}
            </text>
          </g>
        );
      })}
      <text x={startX + NUM_LAYERS * (boxW + gap) + 6} y={190} fontSize={9} fill="var(--ink-muted)">
        ···(共 24 层,同一组权重 W)
      </text>

      <line
        x1={startX + ((activeLayer - 1) * (boxW + gap)) + boxW / 2}
        y1={96}
        x2={startX + boxW / 2}
        y2={170}
        stroke="#10b981"
        strokeWidth={1.4}
        strokeDasharray="3 2"
        markerEnd="url(#arrow-albert)"
      />
      <defs>
        <marker id="arrow-albert" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#10b981" />
        </marker>
      </defs>

      <text x={W / 2} y={260} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
        第 {activeLayer} 层前向计算时:BERT 用自己的第 {activeLayer} 组权重,ALBERT 复用同一组权重 W
      </text>
      <text x={W / 2} y={278} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        参数量:BERT 24 × 12M = 288M → ALBERT 1 × 12M = 12M(少 24×);但 forward 仍需算 24 次,推理时间不变
      </text>
    </svg>
  );
}
