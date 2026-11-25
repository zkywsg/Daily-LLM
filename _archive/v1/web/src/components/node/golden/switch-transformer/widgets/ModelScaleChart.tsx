import { MODEL_SCALE_TABLE } from "../lib/data";

const W = 700;
const H = 360;

// log 尺度柱状图:总参数 vs 激活参数,凸显 Switch-C 1.57T 总参但激活只有 11B。

function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  return `${(n / 1e6).toFixed(0)}M`;
}

export function ModelScaleChart() {
  const PAD = { left: 70, right: 30, top: 50, bottom: 90 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const slot = innerW / MODEL_SCALE_TABLE.length;
  const barW = slot * 0.32;

  // log10 scale, from 1e8 to 2e12
  const logMin = 8;
  const logMax = 12.3;
  const yScale = (v: number) => {
    const lv = Math.log10(Math.max(v, 1));
    return PAD.top + (1 - (lv - logMin) / (logMax - logMin)) * innerH;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Model total params vs active params, log scale">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        总参数 vs 激活参数(log 尺度)— Switch-C 1.57T 总参,激活仍是 11B
      </text>

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {[1e8, 1e9, 1e10, 1e11, 1e12].map((v) => (
        <g key={v}>
          <text x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
            {fmtParams(v)}
          </text>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(v)} y2={yScale(v)} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      {MODEL_SCALE_TABLE.map((row, i) => {
        const x = PAD.left + i * slot + slot / 2;
        const totalY = yScale(row.totalParams);
        const activeY = yScale(row.activeParams);
        const isSwitch = row.name.startsWith("Switch");
        return (
          <g key={row.name}>
            <rect
              x={x - barW - 2}
              y={totalY}
              width={barW}
              height={H - PAD.bottom - totalY}
              rx={2}
              fill={isSwitch ? "#f59e0b" : "#9ca3af"}
              opacity={0.85}
            />
            <rect
              x={x + 2}
              y={activeY}
              width={barW}
              height={H - PAD.bottom - activeY}
              rx={2}
              fill="#3b82f6"
              opacity={0.85}
            />
            <text x={x} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={9} fontWeight={isSwitch ? 700 : 500} fill="var(--ink-primary)">
              {row.name.replace(" (dense)", "")}
            </text>
            <text x={x} y={H - PAD.bottom + 28} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
              ppl {row.c4Perplexity.toFixed(2)}
            </text>
          </g>
        );
      })}

      <rect x={PAD.left} y={H - 34} width={12} height={12} fill="#f59e0b" opacity={0.85} />
      <text x={PAD.left + 18} y={H - 24} fontSize={10} fill="var(--ink-secondary)">总参数(Switch 系列)</text>
      <rect x={PAD.left + 180} y={H - 34} width={12} height={12} fill="#3b82f6" opacity={0.85} />
      <text x={PAD.left + 198} y={H - 24} fontSize={10} fill="var(--ink-secondary)">激活参数</text>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={9} fontStyle="italic" fill="var(--ink-muted)">
        灰/黄柱远高于蓝柱 = 参数容量与激活算力解耦,这是 MoE 的本质买卖
      </text>
    </svg>
  );
}
