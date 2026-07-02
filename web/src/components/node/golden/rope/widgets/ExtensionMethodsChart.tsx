import { EXTENSION_METHODS } from "../lib/data";

const W = 700;
const H = 280;

export function ExtensionMethodsChart() {
  const rowH = 70;
  const startY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RoPE context extension methods">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        长上下文扩展方法 — 都建立在 RoPE 旋转频率结构上
      </text>

      {EXTENSION_METHODS.map((m, i) => {
        const y = startY + i * (rowH + 6);
        return (
          <g key={m.name}>
            <rect x={30} y={y} width={W - 60} height={rowH - 6} rx={4} fill="#fce7f3" fillOpacity={0.15} stroke="#ec4899" strokeOpacity={0.5} strokeWidth={1.2} />
            <text x={44} y={y + 22} fontSize={12} fontWeight={700} fill="#831843">{m.name}</text>
            <text x={44} y={y + 40} fontSize={10} fill="#6b7280">{m.desc}</text>

            <text x={W - 44} y={y + 30} textAnchor="end" fontSize={12} fontWeight={700} fill="#065f46">
              {m.from} → {m.to}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        绝对 PE / learned PE 上几乎不存在等效的"上下文扩展"方法
      </text>
    </svg>
  );
}
