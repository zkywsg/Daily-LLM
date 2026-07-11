import { LM1B_COMPARE } from "../lib/data";

const W = 700;
const H = 360;

// 5 组模型(LSTM baseline / LSTM-Big / MoE-32 / MoE-512 / MoE-2048)的
// 总参数 vs 激活参数双柱对比,log 刻度,直观体现参数与算力"解耦"。

function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 10e9 ? 0 : 1)}B`;
  return `${(n / 1e6).toFixed(0)}M`;
}

export function ParamsVsComputeChart() {
  const PAD = { left: 60, right: 20, top: 56, bottom: 78 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const n = LM1B_COMPARE.length;
  const groupW = innerW / n;
  const barW = groupW * 0.32;

  // log10 scale, params 范围 0.2e9 ~ 137e9
  const logMin = Math.log10(0.15e9);
  const logMax = Math.log10(200e9);
  const yScale = (v: number) => PAD.top + (1 - (Math.log10(v) - logMin) / (logMax - logMin)) * innerH;

  const gridVals = [0.2e9, 1e9, 10e9, 100e9];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="Total params vs active params per token, log scale, across MoE expert counts"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        总参数 vs 每 token 激活参数(log 刻度)— 参数与算力解耦
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        dense 模型(LSTM)两条线重合;MoE 的灰条越堆越高,粉条几乎不变
      </text>

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {gridVals.map((v) => (
        <g key={v}>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(v)} y2={yScale(v)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 6} y={yScale(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
            {fmtParams(v)}
          </text>
        </g>
      ))}

      {LM1B_COMPARE.map((m, i) => {
        const gx = PAD.left + i * groupW;
        const totalY = yScale(m.params);
        const activeY = yScale(m.activeParams);
        return (
          <g key={m.label}>
            <rect
              x={gx + groupW / 2 - barW - 2}
              y={totalY}
              width={barW}
              height={H - PAD.bottom - totalY}
              rx={2}
              fill="#9ca3af"
              opacity={0.82}
            />
            <text x={gx + groupW / 2 - barW / 2 - 2} y={totalY - 4} textAnchor="middle" fontSize={8.5} fontWeight={600} fill="var(--ink-primary)">
              {fmtParams(m.params)}
            </text>

            <rect
              x={gx + groupW / 2 + 2}
              y={activeY}
              width={barW}
              height={H - PAD.bottom - activeY}
              rx={2}
              fill="#ec4899"
              opacity={0.85}
            />
            <text x={gx + groupW / 2 + barW / 2 + 2} y={activeY - 4} textAnchor="middle" fontSize={8.5} fontWeight={600} fill="#831843">
              {fmtParams(m.activeParams)}
            </text>

            {m.label.split("\n").map((line, li) => (
              <text
                key={li}
                x={gx + groupW / 2}
                y={H - PAD.bottom + 16 + li * 11}
                textAnchor="middle"
                fontSize={9}
                fontWeight={500}
                fill="var(--ink-secondary)"
              >
                {line}
              </text>
            ))}
            <text x={gx + groupW / 2} y={H - PAD.bottom + 40} textAnchor="middle" fontSize={8} fontStyle="italic" fill="var(--ink-muted)">
              PPL {m.perplexity.toFixed(1)} · {m.computeX}× 算力
            </text>
          </g>
        );
      })}

      {/* legend */}
      <rect x={PAD.left} y={PAD.top - 22} width={10} height={10} rx={2} fill="#9ca3af" opacity={0.82} />
      <text x={PAD.left + 14} y={PAD.top - 13} fontSize={9} fill="var(--ink-secondary)">总参数</text>
      <rect x={PAD.left + 70} y={PAD.top - 22} width={10} height={10} rx={2} fill="#ec4899" opacity={0.85} />
      <text x={PAD.left + 84} y={PAD.top - 13} fontSize={9} fill="var(--ink-secondary)">每 token 激活参数</text>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        MoE-2048:137B 总参数,每 token 只激活 1.5B(~1%)—— 用 LSTM-Big 1/3 算力拿到更低 perplexity
      </text>
    </svg>
  );
}
