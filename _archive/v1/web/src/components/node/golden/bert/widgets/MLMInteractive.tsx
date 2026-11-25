import type { DemoSentence } from "../lib/data";

interface Props {
  sentence: DemoSentence;
  maskedIdx: number | null;
  onMaskChange: (i: number | null) => void;
}

const W = 700;
const H = 280;
const TOK_Y = 60;
const PRED_Y = 130;

// 点击 token 把它变成 [MASK],右侧弹出 top-k 候选(手工 curated 数据)。
// 让 viewer 摸到"双向上下文如何反推被 mask 词"。
export function MLMInteractive({ sentence, maskedIdx, onMaskChange }: Props) {
  const n = sentence.tokens.length;
  const totalW = W - 60;
  const tokSpacing = totalW / Math.max(1, n);
  const tokX = (i: number) => 30 + tokSpacing * (i + 0.5);
  const candidates = maskedIdx != null ? sentence.predictions[maskedIdx] : null;
  const hasPred = candidates && candidates.length > 0;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="MLM interactive">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        点 token → 把它变成 [MASK],BERT 用双向上下文反推 top-5 候选
      </text>

      {/* token 行 */}
      {sentence.tokens.map((t, i) => {
        const isMasked = i === maskedIdx;
        const hasData = sentence.predictions[i] != null;
        return (
          <g
            key={`tok-${i}`}
            transform={`translate(${tokX(i)}, ${TOK_Y})`}
            style={{ cursor: hasData ? "pointer" : "not-allowed" }}
            onClick={() => hasData && onMaskChange(isMasked ? null : i)}
          >
            <rect
              x={-32}
              y={-16}
              width={64}
              height={32}
              rx={4}
              fill={isMasked ? "#fce7f3" : hasData ? "var(--bg-surface)" : "#f3f4f6"}
              stroke={isMasked ? "#ec4899" : "var(--border)"}
              strokeWidth={isMasked ? 2 : 1}
            />
            <text
              y={4}
              textAnchor="middle"
              fontSize={11}
              fontWeight={isMasked ? 700 : 500}
              fill={hasData ? "var(--ink-primary)" : "var(--ink-muted)"}
            >
              {isMasked ? "[MASK]" : t}
            </text>
            {hasData && !isMasked && (
              <text y={28} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
                可点击
              </text>
            )}
          </g>
        );
      })}

      {/* top-k 预测 */}
      {hasPred && candidates && (
        <g transform={`translate(0, ${PRED_Y})`}>
          <text x={W / 2} y={0} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-secondary)">
            top-5 预测 (越靠左概率越高)
          </text>
          {candidates.map((c, k) => {
            const barW = (W - 100) * (c.prob / candidates[0].prob);
            const y = 18 + k * 22;
            return (
              <g key={`pred-${k}`}>
                <text x={48} y={y + 8} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--ink-primary)">
                  {c.word}
                </text>
                <rect
                  x={56}
                  y={y - 2}
                  width={barW}
                  height={12}
                  rx={2}
                  fill={k === 0 ? "#ec4899" : "#dbeafe"}
                  opacity={k === 0 ? 0.8 : 0.6}
                />
                <text x={56 + barW + 6} y={y + 8} fontSize={10} fill="var(--ink-muted)">
                  {(c.prob * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
        </g>
      )}

      {!hasPred && (
        <text x={W / 2} y={H / 2 + 40} textAnchor="middle" fontSize={11} fill="var(--ink-muted)" fontStyle="italic">
          点击有 "可点击" 提示的 token,看 BERT 怎么用前后文反推它
        </text>
      )}
    </svg>
  );
}
