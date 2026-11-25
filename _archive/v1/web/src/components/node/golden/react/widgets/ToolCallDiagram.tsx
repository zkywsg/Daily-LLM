import { TOOLS } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  selectedIdx: number;
}

export function ToolCallDiagram({ selectedIdx }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ReAct 工具集与调用决策">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Thought 决定调哪个工具、参数怎么填
      </text>

      <rect x={260} y={45} width={180} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={6} />
      <text x={350} y={69} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">Thought(推理目的)</text>

      {TOOLS.map((tool, i) => {
        const x = 30 + i * 165;
        const isSelected = i === selectedIdx;
        return (
          <g key={tool.name} opacity={isSelected ? 1 : 0.4}>
            <line x1={350} y1={85} x2={x + 75} y2={130} stroke={isSelected ? "#ec4899" : "#9ca3af"} strokeWidth={isSelected ? 2 : 1} />
            <rect x={x} y={130} width={150} height={60} fill={isSelected ? "#fce7f3" : "var(--bg-surface)"} stroke={isSelected ? "#ec4899" : "var(--border)"} strokeWidth={isSelected ? 2 : 1} rx={6} />
            <text x={x + 75} y={152} textAnchor="middle" fontSize={9} fontWeight={700} fontFamily="monospace" fill={isSelected ? "#9d174d" : "var(--ink-muted)"}>{tool.name}</text>
            <text x={x + 75} y={168} textAnchor="middle" fontSize={8} fill={isSelected ? "#9d174d" : "var(--ink-muted)"}>{tool.desc.slice(0, 14)}</text>
            <text x={x + 75} y={182} textAnchor="middle" fontSize={8} fill={isSelected ? "#9d174d" : "var(--ink-muted)"}>{tool.desc.slice(14)}</text>
          </g>
        );
      })}

      <text x={W / 2} y={220} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        关键不是工具复杂,而是 LLM 能根据 Thought 决定调哪个工具、参数怎么填
      </text>
    </svg>
  );
}
