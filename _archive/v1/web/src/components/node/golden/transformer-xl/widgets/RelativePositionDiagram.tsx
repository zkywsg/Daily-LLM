interface Props {
  mode: "absolute" | "relative";
}

const W = 700;
const H = 260;

export function RelativePositionDiagram({ mode }: Props) {
  const PAD_L = 60;
  const PAD_T = 50;
  const rowH = 70;
  const cellW = 46;
  const positions = [0, 1, 2, 3, 4, 5];

  const rows = [
    { label: "段 1", y: PAD_T, color: "#3b82f6", bg: "#dbeafe" },
    { label: "段 2", y: PAD_T + rowH, color: "#ec4899", bg: "#fce7f3" },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="绝对位置编码 vs 相对位置编码">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "absolute" ? "绝对 PE — 段 1 和段 2 的位置 5 拿到相同的 PE_5,产生歧义" : "相对 PE — 只编码 query/key 之间的距离 i−j,跨段无歧义"}
      </text>

      {rows.map((row) => (
        <g key={row.label}>
          <text x={PAD_L - 12} y={row.y + 26} textAnchor="end" fontSize={11} fontWeight={700} fill={row.color}>
            {row.label}
          </text>
          {positions.map((p, i) => {
            const x = PAD_L + i * cellW;
            return (
              <g key={p}>
                <rect x={x} y={row.y} width={cellW - 6} height={40} fill={row.bg} stroke={row.color} strokeWidth={1.3} rx={4} />
                <text x={x + (cellW - 6) / 2} y={row.y + 17} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
                  token {p}
                </text>
                <text x={x + (cellW - 6) / 2} y={row.y + 32} textAnchor="middle" fontSize={11} fontWeight={700} fill={row.color}>
                  {mode === "absolute" ? `PE_${p}` : `pos ${p}`}
                </text>
              </g>
            );
          })}
        </g>
      ))}

      {mode === "absolute" ? (
        <>
          {/* 高亮位置 5(index 5) 在两段中的冲突 */}
          <path
            d={`M ${PAD_L + 5 * cellW + (cellW - 6) / 2} ${PAD_T + 40} L ${PAD_L + 5 * cellW + (cellW - 6) / 2} ${PAD_T + rowH}`}
            stroke="#ef4444"
            strokeWidth={1.6}
            strokeDasharray="4 3"
          />
          <text x={PAD_L + 5 * cellW + (cellW - 6) / 2 + 8} y={PAD_T + rowH / 2 + 15} fontSize={9} fill="#ef4444" fontWeight={700}>
            冲突!同为 PE_5
          </text>
        </>
      ) : (
        <text x={W / 2} y={PAD_T + rowH + 60} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          相对 PE 把位置项从"加在输入上"移到"加在 attention score 上":R_{"{i-j}"} 只依赖距离,不依赖绝对索引
        </text>
      )}

      <text x={W / 2} y={PAD_T + rowH + 40} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {mode === "absolute"
          ? "绝对 PE 把位置编码加在 token embedding 上 — 段边界一旦循环拼接,位置信号立刻重复"
          : " "}
      </text>
    </svg>
  );
}
