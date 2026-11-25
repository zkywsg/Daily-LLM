import { QA_EXAMPLES } from "../lib/data";

interface Props {
  exampleIdx: number;
}

const W = 700;
const H = 380;
const COL_W = 320;
const COL_GAP = 20;
const START_X = 20;

// 左右两栏并排:Standard prompt(只给答案)vs CoT prompt(带推理步骤)。
// 在同一道题上,显示模型实际输出 + 是否正确。
// 让 viewer 直接看到"加推理过程后答案正确率不一样"。

function Column({
  x, title, content, finalAnswer, correct, hue,
}: {
  x: number; title: string; content: string[]; finalAnswer: string;
  correct: boolean; hue: number;
}) {
  return (
    <g transform={`translate(${x}, 36)`}>
      <text x={COL_W / 2} y={0} textAnchor="middle" fontSize={13} fontWeight={700} fill={`hsl(${hue}, 60%, 40%)`}>
        {title}
      </text>

      <rect
        x={0}
        y={12}
        width={COL_W}
        height={250}
        rx={5}
        fill={`hsl(${hue}, 70%, 96%)`}
        stroke={`hsl(${hue}, 60%, 70%)`}
        strokeWidth={1.2}
      />

      <text x={12} y={36} fontSize={10} fontWeight={600} fill="#4b5563">
        模型输出:
      </text>
      {content.map((line, i) => (
        <text key={i} x={12} y={56 + i * 18} fontSize={11} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)">
          {line.length > 36 ? line.slice(0, 36) + "…" : line}
        </text>
      ))}

      {/* 最终答案条 */}
      <rect
        x={12}
        y={210}
        width={COL_W - 24}
        height={40}
        rx={4}
        fill={correct ? "#ecfdf5" : "#fef2f2"}
        stroke={correct ? "#10b981" : "#dc2626"}
        strokeWidth={1.5}
      />
      <text x={COL_W / 2} y={228} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
        最终答案
      </text>
      <text x={COL_W / 2} y={244} textAnchor="middle" fontSize={14} fontWeight={700} fill={correct ? "#065f46" : "#7f1d1d"}>
        {finalAnswer}  {correct ? "✓" : "✗"}
      </text>
    </g>
  );
}

export function StandardVsCoTCompare({ exampleIdx }: Props) {
  const ex = QA_EXAMPLES[exampleIdx];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Standard vs CoT prompt comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Q: {ex.question}
      </text>

      <Column
        x={START_X}
        title="Standard prompt"
        content={["A: " + ex.standardAnswer]}
        finalAnswer={ex.standardAnswer}
        correct={ex.standardCorrect}
        hue={220}
      />
      <Column
        x={START_X + COL_W + COL_GAP}
        title="CoT prompt"
        content={[...ex.cotSteps.map((s) => "  " + s), "A: " + ex.cotAnswer]}
        finalAnswer={ex.cotAnswer}
        correct={ex.cotCorrect}
        hue={330}
      />

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        关键:CoT 不是"教模型更多知识",而是"让模型显式把推理过程写出来"
      </text>
    </svg>
  );
}
