import { KV_INJECTION } from "../lib/data";

const W = 680;
const H = 320;

export function PrefixKvInjectionDiagram() {
  const nInput = KV_INJECTION.inputTokens;
  const cellW = 46;
  const cellH = 34;
  const startX = 60;
  const rowGap = 90;

  const rowY = (rowIdx: number) => 60 + rowIdx * rowGap;

  const renderRow = (label: string, rowIdx: number, showPrefix: boolean) => {
    const y = rowY(rowIdx);
    const cells: Array<{ text: string; kind: "prefix" | "input" }> = [];
    if (showPrefix) cells.push({ text: "P", kind: "prefix" });
    for (let i = 1; i <= nInput; i++) cells.push({ text: `${i}`, kind: "input" });

    return (
      <g key={label}>
        <text x={startX - 16} y={y + cellH / 2 + 4} textAnchor="end" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
          {label}
        </text>
        {cells.map((c, i) => {
          const isPrefixBlock = c.kind === "prefix";
          const x = startX + i * cellW;
          return (
            <g key={`${label}-${c.text}-${i}`}>
              <rect
                x={x}
                y={y}
                width={cellW - 6}
                height={cellH}
                rx={5}
                fill={isPrefixBlock ? "#fce7f3" : "#f3f4f6"}
                stroke={isPrefixBlock ? "#ec4899" : "#9ca3af"}
                strokeWidth={isPrefixBlock ? 2 : 1.2}
              />
              <text
                x={x + (cellW - 6) / 2}
                y={y + cellH / 2 + 4}
                textAnchor="middle"
                fontSize={11}
                fontWeight={isPrefixBlock ? 700 : 500}
                fill={isPrefixBlock ? "#be185d" : "#6b7280"}
              >
                {isPrefixBlock ? "P_K/P_V" : c.text}
              </text>
            </g>
          );
        })}
        {showPrefix && (
          <text x={startX} y={y - 8} fontSize={9} fill="#be185d" fontWeight={700}>
            ← m = {KV_INJECTION.prefixLen} 个可学习 prefix token
          </text>
        )}
      </g>
    );
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Prefix Tuning 在 K/V 上的注入方式,Q 不变">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        Prefix 只加在 K / V,不加在 Q
      </text>

      {renderRow("Q(不变)", 0, false)}
      {renderRow("K = [P_K, k₁...kₙ]", 1, true)}
      {renderRow("V = [P_V, v₁...vₙ]", 2, true)}

      <text x={startX} y={H - 14} fontSize={10} fill="var(--ink-muted)" fontStyle="italic">
        ↑ 粉色 = 新增可学习 prefix(每层独立参数);灰色 = 原始 input token 的 K/V,未改动
      </text>
    </svg>
  );
}
