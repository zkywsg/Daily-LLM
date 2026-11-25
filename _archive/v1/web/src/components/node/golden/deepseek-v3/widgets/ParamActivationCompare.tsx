import { PARAM_COMPARE } from "../lib/data";

const W = 700;
const H = 300;

// 总参数 vs 激活参数横向条对比:LLaMA-3.1-405B(dense)/ Mixtral 8x7B / DeepSeek-V3
export function ParamActivationCompare() {
  const PAD = { left: 150, right: 40, top: 50, bottom: 30 };
  const innerW = W - PAD.left - PAD.right;
  const rowH = 60;
  const max = 671; // DeepSeek-V3 总参数,作为满刻度

  const xScale = (v: number) => (v / max) * innerW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="总参数 vs 激活参数对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        总参数(浅色)vs 每 token 激活参数(深色)
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        单位:十亿参数(B)
      </text>

      {PARAM_COMPARE.map((row, i) => {
        const y = PAD.top + i * rowH;
        const totalW = xScale(row.totalParamsB);
        const activeW = xScale(row.activeParamsB);
        const isV3 = row.model === "DeepSeek-V3";
        return (
          <g key={row.model}>
            <text x={PAD.left - 10} y={y + 20} textAnchor="end" fontSize={11} fontWeight={isV3 ? 700 : 500} fill="var(--ink-primary)">
              {row.model}
            </text>
            <rect x={PAD.left} y={y} width={totalW} height={18} rx={3} fill={isV3 ? "#e0e7ff" : "#f3f4f6"} stroke={isV3 ? "#6366f1" : "#9ca3af"} strokeWidth={1} />
            <text x={PAD.left + totalW + 6} y={y + 14} fontSize={10} fill="var(--ink-muted)">
              总 {row.totalParamsB}B
            </text>
            <rect x={PAD.left} y={y + 24} width={activeW} height={18} rx={3} fill={isV3 ? "#6366f1" : "#9ca3af"} />
            <text x={PAD.left + activeW + 6} y={y + 38} fontSize={10} fontWeight={600} fill={isV3 ? "#4338ca" : "var(--ink-secondary)"}>
              激活 {row.activeParamsB}B
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        DeepSeek-V3:671B 总参数,推理只激活 37B —— 容量堪比超大 dense 模型,算力接近小模型
      </text>
    </svg>
  );
}
