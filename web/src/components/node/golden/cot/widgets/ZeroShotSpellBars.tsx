import { SPELL_VARIANTS } from "../lib/data";

const W = 700;
const H = 280;

// Zero-shot CoT 魔法咒语对比:同一模型 + 同一 prompt,只是末尾加一句话,
// GSM8K 准确率从 17.6% → 40.8%。Kojima 2022 的 \"Large Language Models are Zero-Shot Reasoners\"。

export function ZeroShotSpellBars() {
  const maxAcc = Math.max(...SPELL_VARIANTS.map((s) => s.acc));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Zero-shot CoT spell variants accuracy">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Zero-shot CoT — 末尾加不同 \"咒语\",GSM8K 准确率(GPT-3 175B)
      </text>

      {SPELL_VARIANTS.map((s, i) => {
        const y = 50 + i * 50;
        const barW = (W - 420) * (s.acc / maxAcc);
        const isBest = s.acc === maxAcc;
        return (
          <g key={s.spell}>
            <text x={20} y={y + 8} fontSize={11} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)" fontWeight={isBest ? 700 : 500}>
              \"{s.spell}\"
            </text>
            <text x={20} y={y + 22} fontSize={9} fontStyle="italic" fill="var(--ink-muted)">
              {s.note}
            </text>
            <rect
              x={380}
              y={y - 6}
              width={Math.max(2, barW)}
              height={24}
              rx={3}
              fill={`hsl(${s.hue}, 60%, 55%)`}
              opacity={isBest ? 1 : 0.65}
            />
            <text x={385 + barW} y={y + 10} fontSize={11} fontWeight={isBest ? 700 : 500} fill={isBest ? "#831843" : "var(--ink-primary)"}>
              {(s.acc * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        \"Let's think step by step\" 一句话 → 17.6% → 40.8% · 模型一直会推理,只是默认懒得写
      </text>
    </svg>
  );
}
