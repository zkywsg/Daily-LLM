const W = 700;
const H = 300;

interface Props {
  step: number; // 0..N-1, 当前预测哪个位置
}

const SENTENCE = ["The", "detective", "opened", "the", "door", "and", "found"];

// 展示 causal mask: 位置 t 只能看 [0, t-1] 来预测 t
export function AutoregressiveDemo({ step }: Props) {
  const TILE_W = 80;
  const TILE_H = 30;
  const gap = 6;
  const startX = (W - (SENTENCE.length * (TILE_W + gap) - gap)) / 2;
  const y = 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Autoregressive prediction demo">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        自回归预训练 — 预测第 {step + 1} 个词只能看前面 {step} 个
      </text>

      {SENTENCE.map((tok, i) => {
        const x = startX + i * (TILE_W + gap);
        const isVisible = i < step;
        const isTarget = i === step;
        const isFuture = i > step;
        const fill = isTarget ? "#fce7f3" : isVisible ? "#ecfdf5" : "#f3f4f6";
        const stroke = isTarget ? "#ec4899" : isVisible ? "#10b981" : "#d1d5db";
        return (
          <g key={i}>
            <rect x={x} y={y} width={TILE_W} height={TILE_H} rx={4}
                  fill={fill} stroke={stroke} strokeWidth={isTarget ? 2.2 : 1.2}
                  opacity={isFuture ? 0.35 : 1} />
            <text x={x + TILE_W / 2} y={y + TILE_H / 2 + 4}
                  textAnchor="middle" fontSize={11}
                  fontWeight={isTarget ? 700 : 500}
                  fill={isFuture ? "#9ca3af" : "#1f2937"}>
              {isFuture ? "?" : tok}
            </text>
            {isVisible && (
              <line x1={x + TILE_W / 2} y1={y + TILE_H} x2={startX + step * (TILE_W + gap) + TILE_W / 2} y2={y + TILE_H + 30}
                    stroke="#10b981" strokeWidth={1} strokeDasharray="2 2" opacity={0.5} />
            )}
          </g>
        );
      })}

      {/* causal mask indicator */}
      <text x={W / 2} y={y + TILE_H + 55} textAnchor="middle" fontSize={11} fontWeight={600} fill="#065f46">
        {step === 0
          ? "第 1 个词无上文,靠 <s> 起始 token 预测"
          : `Attention 只对前 ${step} 个位置(绿色)计算,后面(灰)被 mask 掉`}
      </text>

      {/* loss formula */}
      <rect x={100} y={190} width={W - 200} height={50} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} opacity={0.6} />
      <text x={W / 2} y={212} textAnchor="middle" fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">
        L_pre = − Σ log p(x_t | x_{"{t-k}"}, ..., x_{"{t-1}"}; θ)
      </text>
      <text x={W / 2} y={230} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#92400e">
        k=512 上下文窗口 · BookCorpus 800M token · 完全无监督
      </text>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        Causal mask 是 decoder-only 自回归建模的根本前提
      </text>
    </svg>
  );
}
