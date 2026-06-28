interface Props {
  withSpell: boolean;
}

const W = 700;
const H = 280;

// 两个 prompt 框对比:加 / 不加 \"Let's think step by step\"。
// 加了之后,模型从 \"直接答\" 变成 \"逐步写推理\"。这是 zero-shot CoT 的本质动作。

export function ZeroShotPromptDemo({ withSpell }: Props) {
  const question = "Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?";
  const baseAns = "A: 8.  (错)";
  const withSpellLines = [
    "A: Let's think step by step.",
    "  - There are 16 balls in total.",
    "  - Half are golf balls → 16 / 2 = 8 golf balls.",
    "  - Half of golf balls are blue → 8 / 2 = 4 blue golf balls.",
    "  - Answer: 4.  ✓",
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Zero-shot CoT prompt ${withSpell ? "with" : "without"} magic spell`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Zero-shot prompt 对比 — {withSpell ? "末尾加咒语" : "原始 prompt"}
      </text>

      {/* prompt 框 */}
      <rect
        x={20}
        y={40}
        width={W - 40}
        height={H - 60}
        rx={6}
        fill={withSpell ? "#fce7f3" : "#dbeafe"}
        stroke={withSpell ? "#ec4899" : "#3b82f6"}
        strokeWidth={1.5}
      />

      {/* question 行 */}
      <text x={32} y={64} fontSize={11} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)" fontWeight={600}>
        {question.length > 70 ? question.slice(0, 70) + "..." : question}
      </text>
      <text x={32} y={80} fontSize={11} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)">
        {question.slice(70, 140)}
      </text>

      {withSpell ? (
        withSpellLines.map((line, i) => (
          <text key={i} x={32} y={108 + i * 22} fontSize={11} fontFamily="ui-monospace, monospace" fill={i === 0 ? "#831843" : "#1f2937"} fontWeight={i === 0 ? 700 : 500}>
            {line}
          </text>
        ))
      ) : (
        <text x={32} y={130} fontSize={13} fontFamily="ui-monospace, monospace" fill="#7f1d1d" fontWeight={700}>
          {baseAns}
        </text>
      )}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {withSpell ? "加咒语 → 模型显式推理 → 正确" : "不加咒语 → 模型偷懒直接猜 → 错"}
      </text>
    </svg>
  );
}
