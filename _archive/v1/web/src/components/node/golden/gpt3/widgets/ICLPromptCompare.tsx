import { ICL_EXAMPLES } from "../lib/scaling";

interface Props {
  exampleIdx: number;
}

const W = 700;
const H = 360;

// 三列 prompt 对比:zero / one / few shot 看同一任务的 prompt 结构变化 + 准确率柱条。
// 让 viewer 摸到"加几个例子就好这么多"是 GPT-3 论文最震撼的图。

const COL_W = 200;
const COL_GAP = 14;
const START_X = 30;

function PromptBlock({
  x,
  title,
  prompt,
  accuracy,
  hue,
}: {
  x: number;
  title: string;
  prompt: string;
  accuracy: number;
  hue: number;
}) {
  const lines = prompt.split("\n");
  return (
    <g transform={`translate(${x}, 40)`}>
      {/* 标题 */}
      <text x={COL_W / 2} y={0} textAnchor="middle" fontSize={12} fontWeight={700} fill={`hsl(${hue}, 60%, 40%)`}>
        {title}
      </text>

      {/* prompt 框 */}
      <rect x={0} y={12} width={COL_W} height={200} rx={5} fill={`hsl(${hue}, 70%, 96%)`} stroke={`hsl(${hue}, 60%, 70%)`} strokeWidth={1.2} />
      {lines.map((line, i) => (
        <text
          key={i}
          x={8}
          y={32 + i * 16}
          fontSize={10}
          fontFamily="ui-monospace, monospace"
          fill="var(--ink-primary)"
        >
          {line.length > 28 ? line.slice(0, 28) + "…" : line}
        </text>
      ))}

      {/* 准确率柱 */}
      <text x={COL_W / 2} y={234} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
        accuracy
      </text>
      <rect x={20} y={244} width={COL_W - 40} height={18} rx={3} fill="var(--bg-surface)" stroke="var(--border)" />
      <rect x={20} y={244} width={(COL_W - 40) * accuracy} height={18} rx={3} fill={`hsl(${hue}, 60%, 55%)`} />
      <text x={COL_W / 2} y={257} textAnchor="middle" fontSize={11} fontWeight={700} fill="#fff">
        {(accuracy * 100).toFixed(0)}%
      </text>
    </g>
  );
}

export function ICLPromptCompare({ exampleIdx }: Props) {
  const ex = ICL_EXAMPLES[exampleIdx];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`In-context learning: ${ex.task}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        任务:{ex.task} — {ex.description}
      </text>

      <PromptBlock x={START_X} title="zero-shot" prompt={ex.zeroShot} accuracy={ex.accuracy.zero} hue={30} />
      <PromptBlock x={START_X + (COL_W + COL_GAP)} title="one-shot" prompt={ex.oneShot} accuracy={ex.accuracy.one} hue={210} />
      <PromptBlock x={START_X + 2 * (COL_W + COL_GAP)} title="few-shot" prompt={ex.fewShot} accuracy={ex.accuracy.few} hue={330} />

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        参数不更新 · 只是在 prompt 里多塞几个示例 → "权重冻结的伪学习"
      </text>
    </svg>
  );
}
