interface Props {
  tokens: string[];
}

const W = 700;
const H = 320;
const PAD = 60;

// 两张 mask 矩阵并排:BERT 全可见(全部填色)vs GPT 下三角因果(右上掩掉)。
// row = query position,col = key position。色块表示 query 可不可以 attend 到该 key。
export function AttentionMaskCompare({ tokens }: Props) {
  const n = tokens.length;
  const matSize = Math.min((W - 3 * PAD) / 2, H - 80);
  const cell = matSize / n;

  const renderMatrix = (ox: number, label: string, isCausal: boolean, color: string) => (
    <g transform={`translate(${ox}, 30)`}>
      <text x={matSize / 2} y={-12} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        {label}
      </text>

      {/* 列标题 (keys) */}
      {tokens.map((t, j) => (
        <text
          key={`c-${j}`}
          x={j * cell + cell / 2}
          y={-2}
          textAnchor="middle"
          fontSize={9}
          fill="var(--ink-muted)"
        >
          {t.length > 4 ? t.slice(0, 4) : t}
        </text>
      ))}

      {/* 行标题 (queries) */}
      {tokens.map((t, i) => (
        <text
          key={`r-${i}`}
          x={-4}
          y={i * cell + cell / 2 + 3}
          textAnchor="end"
          fontSize={9}
          fill="var(--ink-muted)"
        >
          {t.length > 4 ? t.slice(0, 4) : t}
        </text>
      ))}

      {/* 矩阵 cells */}
      {tokens.map((_, i) =>
        tokens.map((__, j) => {
          // BERT:全部可见;GPT:j <= i 才可见
          const visible = isCausal ? j <= i : true;
          return (
            <rect
              key={`${i}-${j}`}
              x={j * cell}
              y={i * cell}
              width={cell}
              height={cell}
              fill={visible ? color : "#1f2937"}
              opacity={visible ? 0.7 : 1}
              stroke="var(--bg-canvas)"
              strokeWidth={0.5}
            />
          );
        }),
      )}

      {/* 注脚 */}
      <text
        x={matSize / 2}
        y={matSize + 22}
        textAnchor="middle"
        fontSize={10}
        fontStyle="italic"
        fill="var(--ink-muted)"
      >
        {isCausal
          ? "GPT:每词只看左侧 (下三角掩掉)"
          : "BERT:每词同时看左右 (全连接)"}
      </text>
    </g>
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT bidirectional vs GPT causal attention mask">
      <text x={W / 2} y={18} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-secondary)">
        粉 = 可 attend · 深灰 = 被 mask 掉
      </text>
      {renderMatrix(PAD, "BERT (encoder-only)", false, "#fce7f3")}
      {renderMatrix(W - PAD - matSize, "GPT (decoder-only)", true, "#dbeafe")}
    </svg>
  );
}
