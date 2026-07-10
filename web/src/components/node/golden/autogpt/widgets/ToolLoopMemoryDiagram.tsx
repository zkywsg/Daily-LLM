import { TOOL_LOOP_STEPS } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  step: number;
}

export function ToolLoopMemoryDiagram({ step }: Props) {
  const current = TOOL_LOOP_STEPS[step];
  const memorySize = (step + 1) * 12;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Tool Loop + Persistent Memory">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Step {step + 1}/{TOOL_LOOP_STEPS.length} — 执行历史存进 vector DB,而不是全塞进 prompt
      </text>

      <rect x={30} y={40} width={640} height={30} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={6} />
      <text x={44} y={60} fontSize={10} fontFamily="monospace" fill="#92400e">task: {current.task}</text>

      <line x1={350} y1={72} x2={350} y2={92} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arrow-loop)" />

      <rect x={30} y={95} width={640} height={30} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={6} />
      <text x={44} y={115} fontSize={10} fontFamily="monospace" fill="#9d174d">tool: {current.tool}()</text>

      <line x1={350} y1={127} x2={350} y2={147} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arrow-loop)" />

      <rect x={30} y={150} width={640} height={30} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={6} />
      <text x={44} y={170} fontSize={10} fill="#1e40af">observation: {current.observation}</text>

      <line x1={350} y1={182} x2={350} y2={202} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arrow-loop)" />

      <rect x={30} y={205} width={640} height={40} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.8} rx={6} />
      <text x={44} y={222} fontSize={10} fontWeight={700} fill="#065f46">Persistent Memory(vector DB)</text>
      <rect x={44} y={228} width={memorySize * 4} height={10} fill="#10b981" rx={2} />
      <text x={44 + memorySize * 4 + 8} y={237} fontSize={9} fill="#065f46">{memorySize} 条记忆(累积)</text>

      <text x={W / 2} y={270} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        需要时检索相关记忆喂回 prompt,而不是把全部历史塞进 context — 这让 agent 能跑几小时/上千步
      </text>

      <defs>
        <marker id="arrow-loop" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
