import { OVER_TRAIN_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  highlightIdx: number;
}

export function OverTrainRatioChart({ highlightIdx }: Props) {
  const PAD_L = 140;
  const PAD_T = 40;
  const plotW = 460;
  const rowH = 48;
  const maxLog = Math.log10(2000);

  const wOf = (r: number) => (Math.log10(r) / maxLog) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="数据/参数比对比(over-train 程度)">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        数据/参数比(log 尺度)— LLaMA 系列持续推高 over-train 程度
      </text>

      {OVER_TRAIN_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        const isChinchilla = row.model === "Chinchilla optimal";
        const color = isChinchilla ? "#9ca3af" : "#10b981";
        const bg = isChinchilla ? "#f3f4f6" : "#ecfdf5";
        return (
          <g key={row.model} opacity={isFocus ? 1 : 0.3}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>{row.model}</text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.ratio), 4)} height={22} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.ratio), 4) + 8} y={y + 16} fontSize={11} fontWeight={700} fill={color}>
              {row.ratio}:1({row.paramsB}B params / {row.tokensB}B tok)
            </text>
          </g>
        );
      })}
    </svg>
  );
}
