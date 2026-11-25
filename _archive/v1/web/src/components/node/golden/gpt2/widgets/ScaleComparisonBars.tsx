import { GPT_SCALING } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  highlightIdx: number;
}

// 6 个模型规模条形(log 参数轴)+ 数据规模条形对比
export function ScaleComparisonBars({ highlightIdx }: Props) {
  const PAD_L = 110;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 30;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const n = GPT_SCALING.length;
  const rowH = plotH / n - 6;

  // log scale for params 100 → 200000 M
  const logMin = 2;
  const logMax = 5.5;
  const wOf = (p: number) => Math.max(((Math.log10(p) - logMin) / (logMax - logMin)) * plotW, 6);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT scaling comparison bars">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        GPT 系列规模演化 — 117M → 175B(log 横轴)
      </text>

      {/* log ticks */}
      {[2, 3, 4, 5].map((lg) => (
        <g key={lg}>
          <line x1={PAD_L + ((lg - logMin) / (logMax - logMin)) * plotW} y1={PAD_T - 4}
                x2={PAD_L + ((lg - logMin) / (logMax - logMin)) * plotW} y2={PAD_T + plotH}
                stroke="#f3f4f6" strokeDasharray="2 3" />
          <text x={PAD_L + ((lg - logMin) / (logMax - logMin)) * plotW} y={PAD_T - 8}
                textAnchor="middle" fontSize={9} fill="#9ca3af">10^{lg}M</text>
        </g>
      ))}

      {GPT_SCALING.map((m, i) => {
        const y = PAD_T + i * (rowH + 6) + 3;
        const w = wOf(m.params);
        const isHighlight = i === highlightIdx;
        const fill = m.isGpt2 ? "#fce7f3" : i === 0 ? "#dbeafe" : "#fef3c7";
        const stroke = m.isGpt2 ? "#ec4899" : i === 0 ? "#3b82f6" : "#f59e0b";
        return (
          <g key={i} opacity={isHighlight || highlightIdx === -1 ? 1 : 0.45}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={isHighlight ? 700 : 500} fill="#374151">{m.name}</text>
            <rect x={PAD_L} y={y} width={w} height={rowH} fill={fill} stroke={stroke} strokeWidth={isHighlight ? 2.2 : 1.2} rx={3} />
            <text x={PAD_L + w + 6} y={y + rowH / 2 + 4} fontSize={10} fontWeight={600} fill="#1f2937">
              {m.params >= 1000 ? (m.params / 1000).toFixed(m.params >= 10000 ? 0 : 1) + "B" : m.params + "M"}
              <tspan dx={6} fill="#6b7280" fontWeight={400}>· {m.layers}L · d={m.dModel} · {m.data}B tok</tspan>
            </text>
          </g>
        );
      })}
    </svg>
  );
}
