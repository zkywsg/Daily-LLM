const W = 700;
const H = 380;

interface Props {
  highlight: "ppo" | "dpo" | "both";
}

function Box({ x, y, w, h, fill, stroke, label, sub }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
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

export function PpoVsDpoPipeline({ highlight }: Props) {
  const dimPpo = highlight === "dpo" ? 0.4 : 1;
  const dimDpo = highlight === "ppo" ? 0.4 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PPO vs DPO pipeline">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        InstructGPT/PPO 三阶段 vs DPO 一阶段
      </text>

      {/* PPO 上行 */}
      <g opacity={dimPpo}>
        <text x={20} y={56} fontSize={12} fontWeight={700} fill="#831843">PPO 路线 · 3 阶段</text>
        <Box x={20}  y={70} w={100} h={50} fill="#fce7f3" stroke="#ec4899" label="① SFT" sub="监督学习" />
        <Arrow x1={120} y1={95} x2={140} y2={95} color="#ec4899" id="ppo1" />
        <Box x={140} y={70} w={130} h={50} fill="#fce7f3" stroke="#ec4899" label="② 训 Reward Model" sub="6B,几万偏好对" />
        <Arrow x1={270} y1={95} x2={290} y2={95} color="#ec4899" id="ppo2" />
        <Box x={290} y={70} w={300} h={50} fill="#fce7f3" stroke="#ec4899" label="③ PPO (actor + critic + RM + π_ref)" sub="4 模型常驻 · 十几个 hparam" />
        <Arrow x1={590} y1={95} x2={610} y2={95} color="#ec4899" id="ppo3" />
        <Box x={610} y={70} w={70} h={50} fill="#ecfdf5" stroke="#10b981" label="π* 对齐" />

        <text x={W / 2} y={140} textAnchor="middle" fontSize={11} fontWeight={600} fill="#831843">
          4 模型 / 14 hparam / ~5000 行 / 几天 · 多节点
        </text>
      </g>

      {/* 分隔 */}
      <line x1={20} y1={170} x2={W - 20} y2={170} stroke="#e5e7eb" />

      {/* DPO 下行 */}
      <g opacity={dimDpo}>
        <text x={20} y={200} fontSize={12} fontWeight={700} fill="#065f46">DPO 路线 · 1 阶段</text>

        <Box x={20}  y={220} w={100} h={50} fill="#ecfdf5" stroke="#10b981" label="SFT (复用)" sub="作为 π_ref" />
        <Arrow x1={120} y1={245} x2={140} y2={245} color="#10b981" id="dpo1" />

        {/* 同一份偏好数据,直接喂进 DPO loss */}
        <Box x={140} y={220} w={170} h={50} fill="#fef3c7" stroke="#f59e0b" label="偏好对 (chosen, rejected)" sub="同 RM 训练所用数据" />
        <Arrow x1={310} y1={245} x2={330} y2={245} color="#f59e0b" id="dpo2" />

        <Box x={330} y={220} w={240} h={50} fill="#ecfdf5" stroke="#10b981" label="DPO Loss (单步训练)" sub="actor + π_ref · 1 hparam (β)" />
        <Arrow x1={570} y1={245} x2={590} y2={245} color="#10b981" id="dpo3" />

        <Box x={590} y={220} w={90} h={50} fill="#ecfdf5" stroke="#10b981" label="π* 对齐" />

        <text x={W / 2} y={290} textAnchor="middle" fontSize={11} fontWeight={600} fill="#065f46">
          2 模型 / 1 hparam (β) / ~300 行 / 几小时 · 单节点 / 成本 1/20
        </text>

        {/* "怎么做到的" */}
        <rect x={140} y={310} width={420} height={40} rx={4} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} opacity={0.6} />
        <text x={W / 2} y={330} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">
          数学等价 · 没有近似:r = β · log(π/π_ref) → BT 偏好 → 直接 cross-entropy
        </text>
      </g>

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        2023.5 Rafailov 把 RM + PPO 折成一个 SFT-style loss · 2024 开源 LLM 默认对齐方法
      </text>
    </svg>
  );
}
