import { chinchillaOptimal } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  flopsLog: number;  // 例如 21, 24, 27 (log10 FLOPs)
}

// 美化数字
function fmtN(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9)  return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6)  return `${(n / 1e6).toFixed(1)}M`;
  return n.toFixed(0);
}

// 估算 H100 训练时间 ( 2 PFLOPS/s × 0.5 效率 )
function trainHours(flops: number, gpus: number): number {
  const throughput = 2e15 * 0.5 * gpus;  // FLOPS/s
  return flops / throughput / 3600;
}

export function ComputeCalculator({ flopsLog }: Props) {
  const flops = Math.pow(10, flopsLog);
  const { N, D } = chinchillaOptimal(flops);

  const PAD = 30;
  const ROW = 40;
  const startY = 70;

  // 一些参考点
  const refs = [
    { name: "GPT-3 175B 训练",  flops: 3.14e23 },
    { name: "Chinchilla 70B",   flops: 5.76e23 },
    { name: "LLaMA-3 8B",        flops: 7.2e23 },
    { name: "GPT-4 (估)",       flops: 2e25 },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Chinchilla compute calculator">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Chinchilla 计算器 — 算力 10^{flopsLog} FLOPs 的最优配置
      </text>

      {/* big output card */}
      <rect x={PAD} y={startY} width={W - PAD * 2} height={ROW * 2 + 8} rx={6}
            fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={PAD + 16} y={startY + 24} fontSize={11} fontWeight={700} fill="#831843" style={{ textTransform: "uppercase" }}>最优配置(N : D = 1 : 20)</text>

      <text x={PAD + 16} y={startY + 52} fontSize={13} fontWeight={700} fill="#1f2937">
        参数: <tspan fontSize={18} fill="#ec4899">{fmtN(N)}</tspan>
      </text>
      <text x={W / 2} y={startY + 52} fontSize={13} fontWeight={700} fill="#1f2937">
        token: <tspan fontSize={18} fill="#ec4899">{fmtN(D)}</tspan>
      </text>
      <text x={PAD + 16} y={startY + 74} fontSize={10} fontStyle="italic" fill="#6b7280">
        基于 C = 6ND · D = 20N → N = √(C / 120)
      </text>

      {/* 训练时长估算 */}
      <text x={PAD} y={startY + ROW * 2 + 30} fontSize={11} fontWeight={700} fill="#374151">
        🖥️ 训练时长(50% MFU H100)
      </text>

      {[100, 1000, 10000].map((g, i) => {
        const hours = trainHours(flops, g);
        const display = hours < 24 ? `${hours.toFixed(1)} h` : `${(hours / 24).toFixed(1)} d`;
        return (
          <text key={i} x={PAD + 16} y={startY + ROW * 2 + 50 + i * 14}
                fontSize={10} fill="#6b7280">
            · {g} GPU: <tspan fontWeight={700} fill="#1f2937">{display}</tspan>
          </text>
        );
      })}

      {/* 参考点列表 */}
      <text x={W / 2 + 30} y={startY + ROW * 2 + 30} fontSize={11} fontWeight={700} fill="#374151">
        🌍 实际模型 FLOPs 参考
      </text>
      {refs.map((r, i) => (
        <text key={i} x={W / 2 + 46} y={startY + ROW * 2 + 50 + i * 14}
              fontSize={10} fill="#6b7280">
          · {r.name}: <tspan fontWeight={700} fill="#1f2937">{r.flops.toExponential(1)}</tspan>
        </text>
      ))}
    </svg>
  );
}
