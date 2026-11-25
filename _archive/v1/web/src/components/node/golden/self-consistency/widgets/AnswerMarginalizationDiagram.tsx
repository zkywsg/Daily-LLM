import { NORMALIZATION_EXAMPLES } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  normalized: boolean;
}

export function AnswerMarginalizationDiagram({ normalized }: Props) {
  const rowH = 48;
  const startY = 60;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="从推理路径中剥离出最终答案,归一化后不同表达方式都归到同一个答案"
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {normalized ? "归一化后:表达方式不同,答案相同" : "未归一化:表达方式不同 → 当成不同答案"}
      </text>

      <text x={40} y={44} fontSize={9} fontWeight={700} fill="var(--ink-muted)" style={{ textTransform: "uppercase" }}>
        推理路径的原始输出
      </text>
      <text x={W - 220} y={44} fontSize={9} fontWeight={700} fill="var(--ink-muted)" style={{ textTransform: "uppercase" }}>
        {normalized ? "归一化后的答案" : "直接投票的答案"}
      </text>

      {NORMALIZATION_EXAMPLES.map((ex, i) => {
        const y = startY + i * rowH;
        const outVal = normalized ? ex.normalized : ex.raw;
        const uniqueColor = normalized ? "#10b981" : ["#3b82f6", "#ec4899", "#f59e0b", "#9ca3af"][i % 4];
        return (
          <g key={i}>
            <rect x={40} y={y} width={330} height={32} rx={6} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} />
            <text x={56} y={y + 20} fontSize={11} fill="#374151">
              "{ex.raw}"
            </text>
            <path
              d={`M 378 ${y + 16} L 470 ${y + 16}`}
              stroke="var(--border)"
              strokeWidth={1.2}
              markerEnd="url(#arrow-sc)"
            />
            <rect x={480} y={y} width={180} height={32} rx={6} fill={`${uniqueColor}22`} stroke={uniqueColor} strokeWidth={1.2} />
            <text x={570} y={y + 20} textAnchor="middle" fontSize={12} fontWeight={700} fill={uniqueColor}>
              {outVal}
            </text>
          </g>
        );
      })}

      <defs>
        <marker id="arrow-sc" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--border)" />
        </marker>
      </defs>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        {normalized
          ? "4 种表达 → 全部归为答案 196,投票时算作 4 票同一答案"
          : "4 种表达 → 被当成 4 个不同答案,投票分裂,谁都不占多数"}
      </text>
    </svg>
  );
}
