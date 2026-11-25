import { GUANACO_SCORES } from "../lib/data";

const W = 700;
const H = 380;

export function GuanacoScoreChart() {
  const PAD_L = 190;
  const PAD_R = 60;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 42;

  const maxVal = 120;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Guanaco 系列 Vicuna Eval 对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        单卡消费级 GPU 训出的 Guanaco-65B,达到 ChatGPT 99.3%
      </text>

      {GUANACO_SCORES.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isGuanaco = row.model.startsWith("Guanaco");
        const isChatgpt = row.model.startsWith("ChatGPT");
        const color = isGuanaco ? "#10b981" : isChatgpt ? "#a855f7" : "#9ca3af";
        const bg = isGuanaco ? "#ecfdf5" : isChatgpt ? "#f3e8ff" : "#f3f4f6";
        return (
          <g key={row.model}>
            <text x={PAD_L - 8} y={y + 14} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>{row.model}</text>
            <text x={PAD_L - 8} y={y + 26} textAnchor="end" fontSize={8} fill="var(--ink-muted)">{row.hardware}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.vicunaScore), 4)} height={20} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.vicunaScore), 4) + 6} y={y + 15} fontSize={10} fontWeight={700} fill={color}>
              {row.vicunaScore}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
