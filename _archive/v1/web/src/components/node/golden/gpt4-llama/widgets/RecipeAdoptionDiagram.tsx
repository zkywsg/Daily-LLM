import { RECIPE_ADOPTERS } from "../lib/data";

const W = 700;
const H = 220;

export function RecipeAdoptionDiagram() {
  const colW = 105;
  const startX = (W - RECIPE_ADOPTERS.length * colW) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="现代配方全行业采用">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Pre-RMSNorm + RoPE + GQA + SwiGLU — 2023 之后几乎全行业采用
      </text>

      <rect x={20} y={40} width={W - 40} height={30} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={6} />
      <text x={W / 2} y={60} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">LLaMA 现代配方(共同基因)</text>

      {RECIPE_ADOPTERS.map((name, i) => {
        const x = startX + i * colW;
        return (
          <g key={name}>
            <line x1={W / 2} y1={70} x2={x + colW / 2 - 4} y2={100} stroke="#9ca3af" strokeWidth={1} />
            <rect x={x} y={100} width={colW - 8} height={34} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={4} />
            <text x={x + (colW - 8) / 2} y={121} textAnchor="middle" fontSize={9} fontWeight={700} fill="#1e40af">{name}</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        差异只在数据组成 / 训练策略 / 后训练,基础架构 2023 之后几乎没再变过
      </text>
    </svg>
  );
}
