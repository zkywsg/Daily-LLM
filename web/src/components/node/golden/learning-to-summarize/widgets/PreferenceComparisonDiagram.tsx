import { CANDIDATE_SUMMARIES } from "../lib/data";

const W = 700;
const H = 360;

interface Props {
  winner: "A" | "B";
}

export function PreferenceComparisonDiagram({ winner }: Props) {
  const boxW = 280;
  const boxH = 130;
  const gap = 40;
  const leftX = W / 2 - gap / 2 - boxW;
  const rightX = W / 2 + gap / 2;
  const boxY = 190;

  const wrap = (text: string, maxChars: number): string[] => {
    const words = text.split("");
    const lines: string[] = [];
    let cur = "";
    for (const ch of words) {
      cur += ch;
      if (cur.length >= maxChars) {
        lines.push(cur);
        cur = "";
      }
    }
    if (cur) lines.push(cur);
    return lines;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="人类偏好比较示意图">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Reddit 帖子 → SFT 模型生成候选摘要 → 标注员选出更好的一个
      </text>

      {/* Source post */}
      <rect x={140} y={36} width={420} height={54} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} />
      <text x={150} y={52} fontSize={9} fontWeight={700} fill="#92400e">原帖(Reddit TL;DR)</text>
      <text x={150} y={68} fontSize={9} fill="#78350f">"换了新工作但通勤变长很多,不确定值不值……" (完整正文省略)</text>
      <text x={150} y={82} fontSize={8} fill="#92400e">→ SFT 模型采样出多个候选摘要</text>

      <line x1={W / 2} y1={90} x2={W / 2} y2={112} stroke="var(--border)" strokeWidth={1.5} markerEnd="url(#arrow-pref)" />
      <defs>
        <marker id="arrow-pref" markerWidth={8} markerHeight={8} refX={4} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--border)" />
        </marker>
      </defs>

      {/* Human labeler icon */}
      <g transform={`translate(${W / 2 - 16}, 118)`}>
        <circle cx={16} cy={10} r={9} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} />
        <path d="M4,34 Q16,14 28,34" fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} />
        <text x={16} y={48} textAnchor="middle" fontSize={8} fill="#374151">标注员</text>
      </g>

      {/* Candidate A */}
      <rect
        x={leftX}
        y={boxY}
        width={boxW}
        height={boxH}
        rx={6}
        fill={winner === "A" ? "#ecfdf5" : "#f3f4f6"}
        stroke={winner === "A" ? "#10b981" : "#9ca3af"}
        strokeWidth={winner === "A" ? 2.4 : 1.2}
      />
      <text x={leftX + 10} y={boxY + 18} fontSize={10} fontWeight={700} fill={winner === "A" ? "#065f46" : "#374151"}>
        候选 A {winner === "A" ? "✓ 被偏好" : ""}
      </text>
      {wrap(CANDIDATE_SUMMARIES[0].text, 24).map((line, i) => (
        <text key={i} x={leftX + 10} y={boxY + 36 + i * 13} fontSize={9} fill="#374151">{line}</text>
      ))}

      {/* Candidate B */}
      <rect
        x={rightX}
        y={boxY}
        width={boxW}
        height={boxH}
        rx={6}
        fill={winner === "B" ? "#ecfdf5" : "#f3f4f6"}
        stroke={winner === "B" ? "#10b981" : "#9ca3af"}
        strokeWidth={winner === "B" ? 2.4 : 1.2}
      />
      <text x={rightX + 10} y={boxY + 18} fontSize={10} fontWeight={700} fill={winner === "B" ? "#065f46" : "#374151"}>
        候选 B {winner === "B" ? "✓ 被偏好" : ""}
      </text>
      {wrap(CANDIDATE_SUMMARIES[1].text, 24).map((line, i) => (
        <text key={i} x={rightX + 10} y={boxY + 36 + i * 13} fontSize={9} fill="#374151">{line}</text>
      ))}

      <text x={W / 2} y={boxY + boxH + 26} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        每个 prompt 生成 4 个候选,标注员两两比较 → 64K 对偏好数据,写作 (x, y_w, y_l)
      </text>
    </svg>
  );
}
