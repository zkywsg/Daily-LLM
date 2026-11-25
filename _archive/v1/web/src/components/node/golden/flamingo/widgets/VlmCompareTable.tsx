import { VLM_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function VlmCompareTable() {
  const rowH = 56;
  const startY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Flamingo vs BLIP-2 vs LLaVA comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Flamingo vs BLIP-2 vs LLaVA — 三条 VLM 路线
      </text>

      <text x={30} y={44} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>模型</text>
      <text x={140} y={44} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>LLM</text>
      <text x={310} y={44} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>桥接</text>
      <text x={560} y={44} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>ICL</text>

      {VLM_COMPARE.map((r, i) => {
        const y = startY + i * (rowH - 4);
        const color = r.name === "Flamingo" ? "#ec4899" : r.name === "BLIP-2" ? "#3b82f6" : "#10b981";
        return (
          <g key={r.name}>
            <rect x={20} y={y} width={W - 40} height={rowH - 10} rx={4}
                  fill={color} fillOpacity={0.06} stroke={color} strokeOpacity={0.4} strokeWidth={1.2} />
            <circle cx={34} cy={y + (rowH - 10) / 2} r={5} fill={color} />
            <text x={44} y={y + (rowH - 10) / 2 + 4} fontSize={11} fontWeight={700} fill={color}>{r.name}</text>

            <text x={140} y={y + (rowH - 10) / 2 - 2} fontSize={10} fill="#374151">{r.llm}</text>
            <text x={140} y={y + (rowH - 10) / 2 + 12} fontSize={9} fill="#6b7280">
              {r.llmSize} · {r.frozen ? "冻结" : "微调"}
            </text>

            <text x={310} y={y + (rowH - 10) / 2 + 4} fontSize={10} fill="#374151">{r.bridge}</text>

            <text x={560} y={y + (rowH - 10) / 2 + 4} fontSize={10} fontWeight={600} fill={color}>{r.icl}</text>
          </g>
        );
      })}
    </svg>
  );
}
