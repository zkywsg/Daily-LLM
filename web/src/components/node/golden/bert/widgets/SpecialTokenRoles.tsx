const W = 700;
const H = 260;

// [CLS] 和 [SEP] 在不同下游任务里扮演的角色 —— 三个 mini 流程图。
// 1) 句子分类:[CLS] 的输出向量过 linear 得分类 logits
// 2) 句对相关性 (NSP / MNLI):[CLS] 同样,但要靠 segment A/B 区分两句
// 3) 序列标注 (NER):每个 token 的输出向量各过 linear

function Box({
  x,
  y,
  w,
  h,
  fill,
  stroke,
  label,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  stroke: string;
  label: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.2} />
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="#1f2937">
        {label}
      </text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  const id = `arr-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} markerEnd={`url(#${id})`} />
    </g>
  );
}

interface Task {
  title: string;
  inputTokens: string[];
  /** 哪些 token 的输出被用来产生预测 */
  outputFrom: number[];
  predictionLabel: string;
}

const TASKS: Task[] = [
  {
    title: "1. 句子分类(SST-2 等)",
    inputTokens: ["[CLS]", "great", "movie"],
    outputFrom: [0],
    predictionLabel: "positive / negative",
  },
  {
    title: "2. 句对相关性(MNLI / NSP)",
    inputTokens: ["[CLS]", "A", "[SEP]", "B"],
    outputFrom: [0],
    predictionLabel: "entail / contradict",
  },
  {
    title: "3. 序列标注(NER)",
    inputTokens: ["John", "from", "NYC"],
    outputFrom: [0, 1, 2],
    predictionLabel: "PER / O / LOC",
  },
];

export function SpecialTokenRoles() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT [CLS] / [SEP] 在不同任务的角色">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        [CLS] / [SEP] 在不同下游任务的角色
      </text>

      {TASKS.map((task, ti) => {
        const ox = 20 + ti * 230;
        return (
          <g key={ti} transform={`translate(${ox}, 38)`}>
            <text x={210 / 2} y={0} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink-secondary)">
              {task.title}
            </text>

            {/* 输入 token row */}
            {task.inputTokens.map((t, i) => {
              const tx = (i / task.inputTokens.length) * 200 + 5;
              const tw = 200 / task.inputTokens.length - 8;
              const isSpecial = t.startsWith("[");
              return (
                <Box
                  key={`in-${i}`}
                  x={tx}
                  y={12}
                  w={tw}
                  h={28}
                  fill={isSpecial ? "#fce7f3" : "#fef3c7"}
                  stroke={isSpecial ? "#ec4899" : "#f59e0b"}
                  label={t}
                />
              );
            })}

            {/* 中间 encoder */}
            <Box x={20} y={64} w={170} h={28} fill="#dbeafe" stroke="#3b82f6" label="BERT encoder" />

            {/* 输出 row,只有 outputFrom 的位置高亮 */}
            {task.inputTokens.map((_, i) => {
              const tx = (i / task.inputTokens.length) * 200 + 5;
              const tw = 200 / task.inputTokens.length - 8;
              const used = task.outputFrom.includes(i);
              return (
                <Box
                  key={`out-${i}`}
                  x={tx}
                  y={108}
                  w={tw}
                  h={22}
                  fill={used ? "#ecfdf5" : "var(--bg-surface)"}
                  stroke={used ? "#10b981" : "var(--border)"}
                  label={used ? "→ pred" : "·"}
                />
              );
            })}

            {/* 箭头 */}
            <Arrow x1={105} y1={42} x2={105} y2={62} />
            <Arrow x1={105} y1={94} x2={105} y2={106} />

            {/* 预测标签 */}
            <text x={105} y={150} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-secondary)">
              {task.predictionLabel}
            </text>
          </g>
        );
      })}

      {/* 注脚 */}
      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        [CLS] 是聚合"整段含义"的占位 token · 它的 encoder 输出最常用作分类信号
      </text>
    </svg>
  );
}
