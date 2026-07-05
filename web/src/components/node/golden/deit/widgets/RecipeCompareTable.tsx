import { RECIPE_ROWS } from "../lib/data";

const W = 700;
const ROW_H = 26;
const H = 50 + RECIPE_ROWS.length * ROW_H;

// 原版 ViT 训练设置 vs DeiT 现代训练 recipe 逐项对比。
export function RecipeCompareTable() {
  const colSetting = 60;
  const colVit = 320;
  const colDeit = 500;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Training recipe comparison table">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        训练设置逐项对比 — 原版 ViT vs DeiT 现代 recipe
      </text>

      <text x={colSetting} y={44} fontSize={10} fontWeight={700} fill="var(--ink-muted)">设置</text>
      <text x={colVit} y={44} fontSize={10} fontWeight={700} fill="var(--ink-muted)">原版 ViT</text>
      <text x={colDeit} y={44} fontSize={10} fontWeight={700} fill="#db2777">DeiT</text>
      <line x1={40} y1={50} x2={W - 40} y2={50} stroke="var(--border)" />

      {RECIPE_ROWS.map((r, i) => {
        const y = 50 + i * ROW_H;
        return (
          <g key={r.setting}>
            {i % 2 === 0 && <rect x={40} y={y} width={W - 80} height={ROW_H} fill="#f9fafb" />}
            <text x={colSetting} y={y + 17} fontSize={10.5} fill="#374151">{r.setting}</text>
            <text x={colVit} y={y + 17} fontSize={10.5} fill="#9ca3af">{r.vit}</text>
            <text x={colDeit} y={y + 17} fontSize={10.5} fontWeight={600} fill="#db2777">{r.deit}</text>
          </g>
        );
      })}
    </svg>
  );
}
