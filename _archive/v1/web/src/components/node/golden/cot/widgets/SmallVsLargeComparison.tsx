const W = 700;
const H = 280;

// 小模型 vs 大模型用 CoT 的行为差异:
//   小模型 CoT:推一步就跑偏(没有足够的"内部推理能力"),反而比 standard 更差
//   大模型 CoT:能稳定推下来,每步都对,显式书写带来增益
// 用两栏卡片呈现典型输出对比。

function Card({
  x, y, w, h, fill, stroke, title, lines, result,
}: {
  x: number; y: number; w: number; h: number;
  fill: string; stroke: string; title: string;
  lines: string[]; result: { text: string; correct: boolean };
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + 22} textAnchor="middle" fontSize={12} fontWeight={700} fill="#1f2937">
        {title}
      </text>
      {lines.map((l, i) => (
        <text key={i} x={x + 14} y={y + 50 + i * 18} fontSize={10} fontFamily="ui-monospace, monospace" fill="#374151">
          {l}
        </text>
      ))}
      <rect x={x + 14} y={y + h - 38} width={w - 28} height={26} rx={3} fill={result.correct ? "#ecfdf5" : "#fef2f2"} stroke={result.correct ? "#10b981" : "#dc2626"} />
      <text x={x + w / 2} y={y + h - 20} textAnchor="middle" fontSize={11} fontWeight={700} fill={result.correct ? "#065f46" : "#7f1d1d"}>
        {result.text}  {result.correct ? "✓" : "✗"}
      </text>
    </g>
  );
}

export function SmallVsLargeComparison() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Small vs large model on CoT">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        小模型 vs 大模型用 CoT 的典型输出对比
      </text>

      <Card
        x={20}
        y={36}
        w={320}
        h={220}
        fill="#fef2f2"
        stroke="#fca5a5"
        title="GPT-3 6.7B + CoT"
        lines={[
          "Q: 23 个苹果...",
          "Let's think step by step:",
          "  - 一开始有 32 个",
          "  - 用 20 个,剩 12 个",
          "  - 加 6 个 = 12 + 6 = 18",
          "(小模型推一步就错)",
        ]}
        result={{ text: "答 18", correct: false }}
      />
      <Card
        x={360}
        y={36}
        w={320}
        h={220}
        fill="#ecfdf5"
        stroke="#86efac"
        title="GPT-3 175B + CoT"
        lines={[
          "Q: 23 个苹果...",
          "Let's think step by step:",
          "  - 一开始有 23 个",
          "  - 用 20 个,剩 23 - 20 = 3",
          "  - 加 6 个 = 3 + 6 = 9",
          "(大模型每步都对)",
        ]}
        result={{ text: "答 9", correct: true }}
      />
    </svg>
  );
}
