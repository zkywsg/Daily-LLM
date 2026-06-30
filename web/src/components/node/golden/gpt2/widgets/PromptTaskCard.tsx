import { PROMPT_TEMPLATES } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  taskIdx: number;
}

// 卡片:展示 prompt 模板 + 例子 input + 例子 output
export function PromptTaskCard({ taskIdx }: Props) {
  const t = PROMPT_TEMPLATES[taskIdx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Zero-shot prompt example">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Zero-shot 任务 — {t.task}
      </text>

      {/* template */}
      <rect x={30} y={50} width={W - 60} height={70} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={42} y={68} fontSize={10} fontWeight={700} fill="#92400e" style={{ textTransform: "uppercase" }}>prompt 模板</text>
      {t.template.split("\n").map((line, i) => (
        <text key={i} x={42} y={88 + i * 14} fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">{line}</text>
      ))}

      {/* example input */}
      <rect x={30} y={140} width={W - 60} height={90} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={42} y={158} fontSize={10} fontWeight={700} fill="#1e40af" style={{ textTransform: "uppercase" }}>送入 GPT-2 的 prompt(具体例子)</text>
      {t.example_in.split("\n").map((line, i) => (
        <text key={i} x={42} y={178 + i * 14} fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">{line}</text>
      ))}

      {/* arrow */}
      <text x={W / 2} y={250} textAnchor="middle" fontSize={11} fontWeight={700} fill="#9ca3af">↓ GPT-2 续写</text>

      {/* output */}
      <rect x={30} y={260} width={W - 60} height={50} rx={6} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
      <text x={42} y={278} fontSize={10} fontWeight={700} fill="#065f46" style={{ textTransform: "uppercase" }}>续写结果(model output)</text>
      <text x={42} y={296} fontSize={12} fontFamily="ui-monospace, monospace" fill="#065f46" fontWeight={700}>{t.example_out}</text>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        没有任何微调 · 没有 task head · 一个 LM + 自然语言 prompt 完成
      </text>
    </svg>
  );
}
