const W = 700;
const H = 320;

// BERT 输入由三层 embedding 相加:Token + Segment + Position。
// 用三排彩色色块叠加,最下面一排是最终输入。
// 让 viewer 看到 [CLS] / [SEP] / Segment A/B 在哪里、怎么编码。

const TOKENS = ["[CLS]", "the", "cat", "[SEP]", "sat", "on", "the", "mat", "[SEP]"];
const SEGMENTS = ["A", "A", "A", "A", "B", "B", "B", "B", "B"];

export function InputCompositionSVG() {
  const n = TOKENS.length;
  const cellW = (W - 80) / n;
  const cellH = 40;
  const rows = [
    { y: 60, label: "Token", colorFor: (i: number) => (TOKENS[i].startsWith("[") ? "#fce7f3" : "#fef3c7") },
    { y: 110, label: "Segment", colorFor: (i: number) => (SEGMENTS[i] === "A" ? "#dbeafe" : "#ecfdf5") },
    { y: 160, label: "Position", colorFor: (i: number) => `hsl(${(i * 35) % 360}, 50%, 88%)` },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT input composition: Token + Segment + Position">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        BERT 输入 = Token emb + Segment emb + Position emb(逐元素相加)
      </text>

      {/* 三排 embedding */}
      {rows.map((row) => (
        <g key={row.label}>
          <text x={70} y={row.y + cellH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--ink-secondary)">
            {row.label}
          </text>
          {TOKENS.map((t, i) => (
            <g key={`${row.label}-${i}`}>
              <rect
                x={80 + i * cellW}
                y={row.y}
                width={cellW - 4}
                height={cellH}
                rx={3}
                fill={row.colorFor(i)}
                stroke="#6b7280"
                strokeWidth={0.8}
              />
              <text
                x={80 + i * cellW + (cellW - 4) / 2}
                y={row.y + cellH / 2 + 4}
                textAnchor="middle"
                fontSize={10}
                fontWeight={500}
                fill="#1f2937"
              >
                {row.label === "Token" ? t : row.label === "Segment" ? `E_${SEGMENTS[i]}` : `E_${i}`}
              </text>
            </g>
          ))}
        </g>
      ))}

      {/* + 号 */}
      <text x={75} y={108} fontSize={18} fontWeight={700} fill="var(--ink-primary)" textAnchor="middle">+</text>
      <text x={75} y={158} fontSize={18} fontWeight={700} fill="var(--ink-primary)" textAnchor="middle">+</text>

      {/* = */}
      <text x={75} y={210} fontSize={18} fontWeight={700} fill="var(--ink-primary)" textAnchor="middle">=</text>

      {/* 求和结果 */}
      <text x={70} y={220 + cellH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        Σ → encoder
      </text>
      {TOKENS.map((_, i) => (
        <rect
          key={`sum-${i}`}
          x={80 + i * cellW}
          y={220}
          width={cellW - 4}
          height={cellH}
          rx={3}
          fill="#fce7f3"
          stroke="#ec4899"
          strokeWidth={1.5}
        />
      ))}

      {/* 注脚 */}
      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        [CLS] 是句对分类信号 · [SEP] 分隔两句 · Segment A/B 让模型知道哪些 token 属于哪句
      </text>
    </svg>
  );
}
