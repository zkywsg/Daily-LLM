import { REACT_TRACE } from "../lib/data";

const W = 700;

interface Props {
  step: number;
}

export function ReactTraceDiagram({ step }: Props) {
  const current = REACT_TRACE[step];
  const H = 260;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ReAct Thought-Action-Observation 循环">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Step {step + 1}/{REACT_TRACE.length} — "Aurora Borealis 是什么颜色?由什么产生?"
      </text>

      <g transform="translate(30, 40)">
        <rect x={0} y={0} width={640} height={50} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.6} rx={6} />
        <text x={14} y={20} fontSize={10} fontWeight={700} fill="#92400e">💭 Thought {step + 1}</text>
        <text x={14} y={38} fontSize={10} fill="#92400e">{current.thought}</text>
      </g>

      <path d="M 350 92 L 350 112" stroke="#9ca3af" strokeWidth={1.6} markerEnd="url(#arrow-react)" />

      <g transform="translate(30, 116)">
        <rect x={0} y={0} width={640} height={40} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.6} rx={6} />
        <text x={14} y={18} fontSize={10} fontWeight={700} fill="#9d174d">🔧 Action {step + 1}</text>
        <text x={14} y={33} fontSize={10} fontFamily="monospace" fill="#9d174d">{current.action}</text>
      </g>

      <path d="M 350 158 L 350 178" stroke="#9ca3af" strokeWidth={1.6} markerEnd="url(#arrow-react)" />

      <g transform="translate(30, 182)">
        <rect x={0} y={0} width={640} height={50} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.6} rx={6} />
        <text x={14} y={20} fontSize={10} fontWeight={700} fill="#065f46">👁 Observation {step + 1}</text>
        <text x={14} y={38} fontSize={9} fill="#065f46">{current.observation.slice(0, 60)}{current.observation.length > 60 ? "..." : ""}</text>
      </g>

      <defs>
        <marker id="arrow-react" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
