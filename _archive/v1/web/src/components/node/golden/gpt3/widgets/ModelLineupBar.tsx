import { MODEL_LINEUP, fmtParams } from "../lib/scaling";

const W = 700;
const H = 240;

// 模型族系条形对比:GPT-1 → GPT-2 → GPT-3 → PaLM → GPT-4。
// 用 log scale 让 117M 那条不会消失。
export function ModelLineupBar() {
  const maxLogParams = Math.log10(Math.max(...MODEL_LINEUP.map((m) => m.params)));
  const minLogParams = Math.log10(Math.min(...MODEL_LINEUP.map((m) => m.params)));
  const barH = 24;
  const gap = 14;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT model lineup parameter comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        模型族系参数对比(log scale,2018→2023)
      </text>

      {MODEL_LINEUP.map((m, i) => {
        const y = 40 + i * (barH + gap);
        const logP = Math.log10(m.params);
        const widthRatio = (logP - minLogParams + 1) / (maxLogParams - minLogParams + 1);
        const barW = widthRatio * (W - 220);
        return (
          <g key={m.name}>
            <text x={14} y={y + barH / 2 + 4} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
              {m.name}
            </text>
            <text x={80} y={y + barH / 2 + 4} fontSize={10} fill="var(--ink-muted)">
              {m.year}
            </text>
            <rect
              x={120}
              y={y}
              width={Math.max(2, barW)}
              height={barH}
              rx={3}
              fill={m.color}
              opacity={0.75}
            />
            <text x={130 + barW} y={y + barH / 2 + 4} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {fmtParams(m.params)}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        GPT-2 → GPT-3 一步 117× · 这是 OpenAI 押 scaling law 直接做出来的
      </text>
    </svg>
  );
}
