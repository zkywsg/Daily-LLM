import { buildCritiqueRevisionFlow, type ConstitutionPrinciple } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  principle: ConstitutionPrinciple;
}

const STAGE_COLORS: Record<string, { bg: string; stroke: string; text: string }> = {
  harmful: { bg: "#fef3c7", stroke: "#f59e0b", text: "#92400e" },
  critique: { bg: "#fce7f3", stroke: "#ec4899", text: "#9d174d" },
  revise: { bg: "#dbeafe", stroke: "#3b82f6", text: "#1e40af" },
  final: { bg: "#ecfdf5", stroke: "#10b981", text: "#065f46" },
};

export function SelfCritiqueRevisionDiagram({ principle }: Props) {
  const flow = buildCritiqueRevisionFlow(principle);
  const boxW = 150;
  const boxH = 90;
  const gap = 20;
  const startX = (W - (flow.length * boxW + (flow.length - 1) * gap)) / 2;
  const y = 70;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="SL-CAI 自我批评与重写流程图">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        SL-CAI:LLM 按 constitution 批评并重写自己的有害回答
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        当前原则:{principle.label} — "{principle.principle}"
      </text>

      {flow.map((step, i) => {
        const x = startX + i * (boxW + gap);
        const colors = STAGE_COLORS[step.stage];
        return (
          <g key={step.stage}>
            <rect x={x} y={y} width={boxW} height={boxH} fill={colors.bg} stroke={colors.stroke} strokeWidth={1.6} rx={6} />
            <text x={x + boxW / 2} y={y + 18} textAnchor="middle" fontSize={9} fontWeight={700} fill={colors.text}>
              {step.label}
            </text>
            {wrapText(step.text, 22).map((line, li) => (
              <text key={li} x={x + boxW / 2} y={y + 38 + li * 12} textAnchor="middle" fontSize={8} fill={colors.text}>
                {line}
              </text>
            ))}
            {i < flow.length - 1 && (
              <path
                d={`M ${x + boxW + 2} ${y + boxH / 2} L ${x + boxW + gap - 2} ${y + boxH / 2}`}
                stroke="var(--ink-muted)"
                strokeWidth={1.6}
                markerEnd="url(#arrow-cai)"
              />
            )}
          </g>
        );
      })}

      <defs>
        <marker id="arrow-cai" markerWidth={8} markerHeight={8} refX={6} refY={3} orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--ink-muted)" />
        </marker>
      </defs>

      <text x={W / 2} y={y + boxH + 40} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        整个流程不需要人类介入 — critique 与 revise 都由同一个 LLM 完成,通常对每个 prompt 反复迭代 4 轮
      </text>
    </svg>
  );
}

function wrapText(text: string, maxChars: number): string[] {
  const words = text.split(" ");
  const lines: string[] = [];
  let current = "";
  for (const w of words) {
    if ((current + " " + w).trim().length > maxChars) {
      lines.push(current.trim());
      current = w;
    } else {
      current = (current + " " + w).trim();
    }
  }
  if (current) lines.push(current);
  return lines.slice(0, 4);
}
