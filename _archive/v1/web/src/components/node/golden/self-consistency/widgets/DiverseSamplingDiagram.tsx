import { SHEEP_MILK_PATHS } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  numSamples: number;
}

export function DiverseSamplingDiagram({ numSamples }: Props) {
  const paths = SHEEP_MILK_PATHS.slice(0, numSamples);
  const colW = W / Math.max(paths.length, 1);
  const votes196 = paths.filter((p) => p.answer === "196").length;
  const votes168 = paths.filter((p) => p.answer === "168").length;
  const winner = votes196 >= votes168 ? "196" : "168";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="温度采样出 N 条不同推理路径,各自得到答案,多数投票选出最终答案"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        温度采样(T=0.7)N={paths.length} 条独立 CoT 路径
      </text>

      <rect x={W / 2 - 90} y={34} width={180} height={26} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} />
      <text x={W / 2} y={51} textAnchor="middle" fontSize={11} fill="#92400e">
        问题:农场羊奶(7 只 × 4 升/天)
      </text>

      {paths.map((p, i) => {
        const cx = colW * i + colW / 2;
        const boxColor = p.correct ? "#dbeafe" : "#fce7f3";
        const strokeColor = p.correct ? "#3b82f6" : "#ec4899";
        return (
          <g key={p.id}>
            <line x1={W / 2} y1={60} x2={cx} y2={100} stroke="var(--border)" strokeWidth={1} />
            <rect x={cx - colW / 2 + 8} y={100} width={colW - 16} height={92} rx={6} fill={boxColor} stroke={strokeColor} strokeWidth={1.2} />
            <text x={cx} y={116} textAnchor="middle" fontSize={9} fontWeight={700} fill="var(--ink-primary)">
              路径 {p.id}
            </text>
            <foreignObject x={cx - colW / 2 + 12} y={122} width={colW - 24} height={58}>
              <div style={{ fontSize: 8, lineHeight: 1.3, color: "#374151", fontFamily: "system-ui" }}>
                {p.reasoning}
              </div>
            </foreignObject>
            <line x1={cx} y1={192} x2={cx} y2={220} stroke={strokeColor} strokeWidth={1.2} />
            <rect x={cx - 28} y={220} width={56} height={24} rx={5} fill={strokeColor} opacity={0.15} stroke={strokeColor} />
            <text x={cx} y={236} textAnchor="middle" fontSize={12} fontWeight={700} fill={strokeColor}>
              {p.answer}
            </text>
            <line x1={cx} y1={244} x2={W / 2} y2={272} stroke={strokeColor} strokeWidth={1} strokeDasharray="3,2" />
          </g>
        );
      })}

      <rect x={W / 2 - 100} y={272} width={200} height={44} rx={8} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
      <text x={W / 2} y={290} textAnchor="middle" fontSize={10} fontWeight={700} fill="#065f46">
        多数投票 → 答案 {winner}
      </text>
      <text x={W / 2} y={305} textAnchor="middle" fontSize={9} fill="#065f46">
        {votes196}/{paths.length} 支持 196{votes168 > 0 ? ` · ${votes168}/${paths.length} 支持 168` : ""}
      </text>

      <text x={W / 2} y={330} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        N 越大,路径越多样,错误路径(粉)越容易被正确路径(蓝)淹没
      </text>
    </svg>
  );
}
