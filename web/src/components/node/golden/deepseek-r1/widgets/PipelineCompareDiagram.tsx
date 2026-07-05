const W = 700;
const H = 220;

export function PipelineCompareDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="R1-Zero vs R1 训练流程对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        两条路径,共享同一个 RL 内核(GRPO)
      </text>

      <g transform="translate(30, 40)">
        <text x={0} y={0} fontSize={10} fontWeight={700} fill="#3b82f6">R1-Zero</text>
        <rect x={0} y={10} width={100} height={36} fill="#dbeafe" stroke="#3b82f6" rx={4} />
        <text x={50} y={32} textAnchor="middle" fontSize={10} fontWeight={700} fill="#1e40af">DeepSeek-V3</text>

        <line x1={100} y1={28} x2={150} y2={28} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-r1)" />

        <rect x={150} y={10} width={110} height={36} fill="#ecfdf5" stroke="#10b981" strokeWidth={2} rx={4} />
        <text x={205} y={32} textAnchor="middle" fontSize={10} fontWeight={700} fill="#065f46">GRPO(纯 RL)</text>

        <line x1={260} y1={28} x2={310} y2={28} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-r1)" />

        <rect x={310} y={10} width={90} height={36} fill="#fef3c7" stroke="#f59e0b" rx={4} />
        <text x={355} y={32} textAnchor="middle" fontSize={10} fontWeight={700} fill="#b45309">R1-Zero</text>

        <text x={420} y={32} fontSize={9} fill="var(--ink-muted)">AIME 15.6% → 71.0%,完全跳过 SFT</text>
      </g>

      <g transform="translate(30, 110)">
        <text x={0} y={0} fontSize={10} fontWeight={700} fill="#ec4899">R1</text>
        <rect x={0} y={10} width={100} height={36} fill="#dbeafe" stroke="#3b82f6" rx={4} />
        <text x={50} y={32} textAnchor="middle" fontSize={10} fontWeight={700} fill="#1e40af">DeepSeek-V3</text>

        <line x1={100} y1={28} x2={140} y2={28} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-r1)" />

        {["冷启动 SFT", "推理 RL", "拒绝采样 SFT", "RLHF"].map((label, i) => (
          <g key={i} transform={`translate(${140 + i * 95}, 0)`}>
            <rect x={0} y={10} width={85} height={36} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={4} />
            <text x={42} y={26} textAnchor="middle" fontSize={8} fontWeight={700} fill="#be185d">{label}</text>
            <text x={42} y={38} textAnchor="middle" fontSize={7} fill="#be185d">Stage {i + 1}</text>
            {i < 3 && <line x1={85} y1={28} x2={95} y2={28} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#arrow-r1)" />}
          </g>
        ))}
      </g>

      <text x={30} y={190} fontSize={10} fill="var(--ink-muted)">
        R1 在 R1-Zero 经验上加 4 阶段训练,把推理能力 + 通用能力 + 对齐都装进同一个模型,AIME 达 79.8%,与 o1 持平
      </text>

      <defs>
        <marker id="arrow-r1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
