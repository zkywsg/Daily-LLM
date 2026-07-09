import { CANDIDATE_EXAMPLES } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  idx: number;
}

export function CandidateSamplingDiagram({ idx }: Props) {
  const ex = CANDIDATE_EXAMPLES[idx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LM 自采样候选 API 调用位置">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LLM 看 ≤20 个手写例子后,自己决定该不该插调用
      </text>

      <rect x={30} y={45} width={640} height={40} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={6} />
      <text x={50} y={70} fontSize={12} fontFamily="monospace" fill="#374151">{ex.sentence}</text>

      <line x1={350} y1={90} x2={350} y2={115} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arrow-tf)" />

      {ex.probAboveThreshold ? (
        <>
          <rect x={30} y={120} width={640} height={50} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.8} rx={6} />
          <text x={50} y={142} fontSize={11} fontWeight={700} fill="#065f46">✓ P(插入调用) &gt; 阈值 τ — 采样候选</text>
          <text x={50} y={160} fontSize={11} fontFamily="monospace" fill="#065f46">{ex.call}</text>
        </>
      ) : (
        <>
          <rect x={30} y={120} width={640} height={50} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.8} rx={6} />
          <text x={50} y={142} fontSize={11} fontWeight={700} fill="#9d174d">✗ P(插入调用) &lt; 阈值 τ — 不采样</text>
          <text x={50} y={160} fontSize={10} fill="#9d174d">这句话本身足够简单,不需要工具辅助</text>
        </>
      )}

      <text x={W / 2} y={210} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        整个语料百万级,人工标不动;但 LLM 自己 prompt 自己,几乎零成本生成海量候选
      </text>

      <defs>
        <marker id="arrow-tf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
