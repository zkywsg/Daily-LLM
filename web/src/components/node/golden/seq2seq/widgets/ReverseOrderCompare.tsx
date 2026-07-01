import { ORDER_COMPARE } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  reversed: boolean;
}

export function ReverseOrderCompare({ reversed }: Props) {
  const data = ORDER_COMPARE[reversed ? 1 : 0];
  const TILE_W = 90;
  const gap = 10;
  const startX = (W - (data.tokens.length * (TILE_W + gap) - gap)) / 2;
  const y = 90;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Reversed input order comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {reversed ? "倒序输入 — x_1 离 c 只有 1 步" : "正序输入 — x_1 离 c 有 T 步"}
      </text>

      {data.tokens.map((tok, i) => (
        <g key={i}>
          <rect x={startX + i * (TILE_W + gap)} y={y} width={TILE_W} height={36} rx={4}
                fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} />
          <text x={startX + i * (TILE_W + gap) + TILE_W / 2} y={y + 23}
                textAnchor="middle" fontSize={13} fontWeight={600} fill="#1f2937">
            {tok}
          </text>
          {i < data.tokens.length - 1 && (
            <line x1={startX + (i + 1) * (TILE_W + gap) - gap} y1={y + 18}
                  x2={startX + (i + 1) * (TILE_W + gap)} y2={y + 18}
                  stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#ro-arr)" />
          )}
        </g>
      ))}
      <defs>
        <marker id="ro-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* c at end */}
      <line x1={startX + data.tokens.length * (TILE_W + gap) - gap} y1={y + 18}
            x2={startX + data.tokens.length * (TILE_W + gap) + 30} y2={y + 18}
            stroke="#f59e0b" strokeWidth={1.4} markerEnd="url(#ro-arr2)" />
      <defs>
        <marker id="ro-arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
        </marker>
      </defs>
      <circle cx={startX + data.tokens.length * (TILE_W + gap) + 55} cy={y + 18} r={26}
              fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
      <text x={startX + data.tokens.length * (TILE_W + gap) + 55} y={y + 23}
            textAnchor="middle" fontSize={13} fontWeight={700} fill="#92400e">c</text>

      {/* 距离标注 */}
      {data.tokens.map((tok, i) => (
        <text key={i} x={startX + i * (TILE_W + gap) + TILE_W / 2} y={y + 60}
              textAnchor="middle" fontSize={11} fontWeight={700}
              fill={data.distanceToC[i] === 1 ? "#065f46" : "#9ca3af"}>
          距 c: {data.distanceToC[i]} 步
        </text>
      ))}

      <rect x={100} y={190} width={W - 200} height={50} rx={6}
            fill={reversed ? "#ecfdf5" : "#fce7f3"}
            stroke={reversed ? "#10b981" : "#ec4899"} strokeWidth={1.2} />
      <text x={W / 2} y={212} textAnchor="middle" fontSize={11} fontWeight={700}
            fill={reversed ? "#065f46" : "#831843"}>
        {reversed
          ? "第一个词 x_1 = \"I\" 距 c 仅 1 步 — 梯度路径最短"
          : "第一个词 x_1 = \"I\" 距 c 有 T=3 步 — 梯度要走完整序列"}
      </text>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        倒序让源句开头和目标句开头之间的"梯度路径"更短,BLEU +4-5(attention 出现后此 trick 被淘汰)
      </text>
    </svg>
  );
}
