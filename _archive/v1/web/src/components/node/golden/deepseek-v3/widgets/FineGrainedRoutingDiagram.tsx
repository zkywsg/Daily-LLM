import { GRANULARITY_COMPARE } from "../lib/data";

interface Props {
  granularity: "coarse" | "fine";
}

const W = 700;
const H = 380;

// 粗粒度(Mixtral 8 expert,top-2)vs 细粒度(V3 256 expert + 1 shared,top-8)。
// 用网格展示"选中"的 expert 数量与网格密度对比,shared expert 单独画在网格外。

export function FineGrainedRoutingDiagram({ granularity }: Props) {
  const row = GRANULARITY_COMPARE[granularity === "coarse" ? 0 : 1];
  const cols = granularity === "coarse" ? 8 : 16;
  const rows = granularity === "coarse" ? 1 : 16;
  const topK = granularity === "coarse" ? 2 : 8;
  const total = cols * rows;

  const gridLeft = 60;
  const gridTop = 90;
  const gridW = 420;
  const gridH = granularity === "coarse" ? 40 : 240;
  const cellW = gridW / cols;
  const cellH = gridH / rows;

  // 确定性选出 topK 个 cell 作为 "选中"(演示用)
  const selected = new Set<number>();
  let seed = granularity === "coarse" ? 7 : 41;
  while (selected.size < topK) {
    seed = (seed * 137 + 91) % total;
    selected.add(seed);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`${row.model} 专家粒度示意`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {row.model} — 每层 {row.expertsPerLayer} 个 expert,top-{row.topK}
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        每 expert ≈ {row.expertSize} 参数 · 激活总参数 ≈ {row.activeExperts}
      </text>

      {/* routed experts 网格 */}
      {Array.from({ length: total }, (_, i) => {
        const c = i % cols;
        const r = Math.floor(i / cols);
        const isSel = selected.has(i);
        return (
          <rect
            key={i}
            x={gridLeft + c * cellW + 1}
            y={gridTop + r * cellH + 1}
            width={cellW - 2}
            height={cellH - 2}
            rx={1}
            fill={isSel ? "#ec4899" : "#fce7f3"}
            stroke={isSel ? "#be185d" : "#f3d3e6"}
            strokeWidth={isSel ? 1.2 : 0.6}
            opacity={isSel ? 1 : 0.7}
          />
        );
      })}

      <text x={gridLeft} y={gridTop - 10} fontSize={10} fontWeight={600} fill="var(--ink-secondary)">
        routed experts({total} 个,粉色 = 本次被选中的 top-{topK})
      </text>

      {/* shared expert */}
      <g transform={`translate(${gridLeft + gridW + 60}, ${gridTop})`}>
        <rect x={0} y={0} width={90} height={granularity === "coarse" ? 40 : 90} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.6} />
        <text x={45} y={granularity === "coarse" ? 24 : 40} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1d4ed8">
          shared
        </text>
        <text x={45} y={granularity === "coarse" ? 24 + 14 : 40 + 16} textAnchor="middle" fontSize={10} fill="#1d4ed8">
          expert
        </text>
        <text x={45} y={granularity === "coarse" ? -8 : -10} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
          {granularity === "coarse" ? "(Mixtral 无)" : "所有 token 都过"}
        </text>
      </g>

      <text x={W / 2} y={H - 24} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {granularity === "coarse"
          ? "8 选 2:组合空间 C(8,2) = 28 种"
          : "256 选 8:组合空间 C(256,8) ≈ 10¹³ 种,专精度远高于粗粒度"}
      </text>
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        细粒度 expert 更小、更专精;shared expert 承担通用能力,routed expert 才能更专注特化
      </text>
    </svg>
  );
}
