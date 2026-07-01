import { OLD_PARADIGM_TASKS } from "../lib/data";

const W = 700;
const H = 380;

interface Props {
  side: "old" | "new" | "both";
}

function Box({ x, y, w, h, fill, stroke, label, sub }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function ParadigmShift({ side }: Props) {
  const dimOld = side === "new" ? 0.35 : 1;
  const dimNew = side === "old" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Old paradigm vs GPT-1 paradigm">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        旧范式:每任务专门模型 vs GPT-1:预训练 + 微调
      </text>

      {/* === 旧范式 === */}
      <g opacity={dimOld}>
        <text x={20} y={50} fontSize={12} fontWeight={700} fill="#831843">旧范式 · 每任务从头训</text>
        {OLD_PARADIGM_TASKS.map((t, i) => {
          const y = 60 + i * 44;
          return (
            <g key={i}>
              <Box x={20} y={y} w={110} h={34} fill="#fce7f3" stroke="#ec4899" label={t.task} sub={t.dataSize} />
              <Arrow x1={130} y1={y + 17} x2={170} y2={y + 17} color="#ec4899" id={`old-a-${i}`} />
              <Box x={170} y={y} w={110} h={34} fill="#fef3c7" stroke="#f59e0b" label={t.oldModel} sub="从零训练" />
            </g>
          );
        })}
        <text x={150} y={250} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#831843">
          换任务 = 换数据 + 换网络 + 换超参
        </text>
        <text x={150} y={268} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#9ca3af">
          TB 级无标注文本浪费 &gt; 99%
        </text>
      </g>

      {/* 分隔 */}
      <line x1={330} y1={40} x2={330} y2={290} stroke="#e5e7eb" strokeDasharray="3 3" />

      {/* === GPT-1 新范式 === */}
      <g opacity={dimNew}>
        <text x={510} y={50} textAnchor="middle" fontSize={12} fontWeight={700} fill="#065f46">GPT-1 · 预训练 + 微调</text>

        <Box x={370} y={70} w={280} h={44} fill="#dbeafe" stroke="#3b82f6"
             label="① 预训练:BookCorpus 800M token" sub="decoder-only Transformer 117M · 自回归 LM" />
        <Arrow x1={510} y1={114} x2={510} y2={140} color="#3b82f6" id="new-a" />

        <Box x={370} y={140} w={280} h={44} fill="#ecfdf5" stroke="#10b981"
             label="② 微调:同一 Transformer + linear head" sub="4 类任务统一序列格式" />

        {OLD_PARADIGM_TASKS.map((t, i) => {
          const x = 370 + i * 70;
          return (
            <g key={i}>
              <Arrow x1={510} y1={184} x2={x + 30} y2={210} color="#10b981" id={`new-t-${i}`} />
              <Box x={x} y={210} w={62} h={30} fill="#fce7f3" stroke="#ec4899" label={t.task} />
            </g>
          );
        })}

        <text x={510} y={270} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#065f46">
          一个模型 · 只换最后一层 head
        </text>
        <text x={510} y={288} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#9ca3af">
          12 个 benchmark 拿 9 个 SOTA
        </text>
      </g>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        统一 backbone + 统一接口让 NLP 从"专门模型时代"进入"预训练时代"
      </text>
    </svg>
  );
}
