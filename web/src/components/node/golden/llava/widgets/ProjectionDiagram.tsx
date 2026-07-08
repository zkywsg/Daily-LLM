const W = 700;
const H = 240;

interface Props {
  mode: "linear" | "mlp";
}

export function ProjectionDiagram({ mode }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CLIP 特征投影到 LLM 空间">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "linear" ? "LLaVA-1.0:单层 Linear Projection" : "LLaVA-1.5:2 层 MLP Projection"}
      </text>

      <rect x={30} y={70} width={130} height={60} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={6} />
      <text x={95} y={96} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">CLIP 特征</text>
      <text x={95} y={114} textAnchor="middle" fontSize={10} fill="#1e40af">768 维</text>

      <line x1={160} y1={100} x2={210} y2={100} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-proj)" />

      {mode === "linear" ? (
        <rect x={210} y={80} width={160} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} rx={6} />
      ) : (
        <>
          <rect x={210} y={80} width={70} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} rx={6} />
          <rect x={300} y={80} width={70} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} rx={6} />
        </>
      )}
      <text x={mode === "linear" ? 290 : 245} y={104} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">Linear</text>
      {mode === "mlp" && <>
        <text x={335} y={104} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">GELU+Linear</text>
      </>}

      <line x1={mode === "linear" ? 370 : 370} y1={100} x2={420} y2={100} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-proj)" />

      <rect x={420} y={70} width={200} height={60} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={6} />
      <text x={520} y={96} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">LLM Token Embedding</text>
      <text x={520} y={114} textAnchor="middle" fontSize={10} fill="#065f46">4096/5120 维(直接拼接)</text>

      <text x={W / 2} y={170} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {mode === "linear"
          ? "唯一从零训练的组件,~4M 参数,相比 LLaMA-7B 几乎可忽略"
          : "表达力更强,LLaVA-1.5 用 2 层 MLP 替代单 Linear,+1.4 分"}
      </text>

      <defs>
        <marker id="arrow-proj" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
