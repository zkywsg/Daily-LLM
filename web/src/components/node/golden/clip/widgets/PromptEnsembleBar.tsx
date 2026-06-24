import { PROMPT_TEMPLATES } from "../lib/data";

const W = 700;
const H = 240;

// Prompt template 对比:同一个模型,只换 prompt 措辞,ImageNet zero-shot 精度差好几个点。
// 这是 CLIP 论文里 prompt engineering 的关键发现 —— 也是后来 zero-shot LLM
// 高度依赖 prompt 的源头。

export function PromptEnsembleBar() {
  const maxAcc = Math.max(...PROMPT_TEMPLATES.map((p) => p.accuracy));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Prompt template accuracy comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        ImageNet zero-shot 精度 — prompt 措辞影响 5~10 个点
      </text>

      {PROMPT_TEMPLATES.map((p, i) => {
        const y = 50 + i * 42;
        const barW = (W - 380) * (p.accuracy / maxAcc);
        const isBest = p.accuracy === maxAcc;
        return (
          <g key={p.template}>
            <text x={20} y={y + 10} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
              {p.template}
            </text>
            <text x={20} y={y + 24} fontSize={9} fill="var(--ink-muted)" fontStyle="italic">
              {p.note}
            </text>
            <rect
              x={320}
              y={y - 4}
              width={Math.max(2, barW)}
              height={20}
              rx={3}
              fill={isBest ? "#10b981" : "#dbeafe"}
              opacity={isBest ? 1 : 0.6}
            />
            <text
              x={325 + barW}
              y={y + 10}
              fontSize={11}
              fontWeight={isBest ? 700 : 500}
              fill={isBest ? "#065f46" : "var(--ink-primary)"}
            >
              {(p.accuracy * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        ImageNet zero-shot benchmark · CLIP 论文 Table 8 + §3.1.4
      </text>
    </svg>
  );
}
