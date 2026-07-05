import { RECIPE_TABLE } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  highlightIdx: number;
}

export function RecipeEvolutionDiagram({ highlightIdx }: Props) {
  const rowH = 44;
  const colW = 190;
  const startX = 20;
  const startY = 50;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="现代 LLM 配方 6 件套演化">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Transformer(2017)→ GPT-3(2020)→ LLaMA(2023)组件替换
      </text>

      <text x={startX} y={startY - 10} fontSize={9} fontWeight={700} fill="#6b7280">组件</text>
      <text x={startX + colW} y={startY - 10} fontSize={9} fontWeight={700} fill="#9ca3af">2017</text>
      <text x={startX + colW * 2} y={startY - 10} fontSize={9} fontWeight={700} fill="#3b82f6">GPT-3</text>
      <text x={startX + colW * 3} y={startY - 10} fontSize={9} fontWeight={700} fill="#10b981">LLaMA</text>

      {RECIPE_TABLE.map((row, i) => {
        const y = startY + i * rowH;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        return (
          <g key={row.component} opacity={isFocus ? 1 : 0.3}>
            <text x={startX} y={y + 14} fontSize={10} fontWeight={700} fill="var(--ink-primary)">{row.component}</text>
            <text x={startX + colW} y={y + 14} fontSize={9} fill="#9ca3af">{row.original2017}</text>
            <text x={startX + colW * 2} y={y + 14} fontSize={9} fill="#3b82f6">{row.gpt3}</text>
            <rect x={startX + colW * 3 - 6} y={y} width={colW} height={20} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} rx={3} opacity={isFocus ? 1 : 0} />
            <text x={startX + colW * 3} y={y + 14} fontSize={9} fontWeight={700} fill="#065f46">{row.llama}</text>
          </g>
        );
      })}
    </svg>
  );
}
