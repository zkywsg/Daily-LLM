interface Props {
  /** 0-2 高亮当前阶段 */
  activeStage: 0 | 1 | 2;
}

const W = 720;
const H = 280;

// SFT → RM → PPO 三阶段总览,activeStage 决定高亮哪一段。
// 用浅色 box + 当前阶段加粗边框 + 弧线连接。

function Box({
  x, y, w, h, fill, stroke, label, sub, highlight,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string;
  label: string; sub: string; highlight?: boolean;
}) {
  return (
    <g opacity={highlight === false ? 0.5 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={fill} stroke={stroke} strokeWidth={highlight ? 3 : 1.5} />
      <text x={x + w / 2} y={y + 24} textAnchor="middle" fontSize={13} fontWeight={700} fill="#1f2937">{label}</text>
      <text x={x + w / 2} y={y + 42} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, label, faded }: { x1: number; y1: number; x2: number; y2: number; label?: string; faded?: boolean }) {
  const id = `pipe-${x1}-${y1}-${x2}-${y2}`;
  return (
    <g opacity={faded ? 0.4 : 1}>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.6} markerEnd={`url(#${id})`} />
      {label && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#6b7280">
          {label}
        </text>
      )}
    </g>
  );
}

const STAGE_COLOR = ["#fef3c7", "#fce7f3", "#dbeafe"];
const STAGE_STROKE = ["#f59e0b", "#ec4899", "#3b82f6"];

export function RLHFPipelineFlow({ activeStage }: Props) {
  const stageW = 180;
  const stageH = 70;
  const gap = 40;
  const startX = 30;
  const y = 90;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`RLHF pipeline, stage ${activeStage + 1}`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        InstructGPT RLHF pipeline — 三阶段顺序训练
      </text>

      {/* Stage 1: SFT */}
      <Box x={startX} y={y} w={stageW} h={stageH} fill={STAGE_COLOR[0]} stroke={STAGE_STROKE[0]} label="① SFT" sub="人类示范 → fine-tune GPT-3" highlight={activeStage === 0} />
      <Arrow x1={startX + stageW} y1={y + stageH / 2} x2={startX + stageW + gap} y2={y + stageH / 2} faded={activeStage !== 0 && activeStage !== 1} />

      {/* Stage 2: RM */}
      <Box x={startX + stageW + gap} y={y} w={stageW} h={stageH} fill={STAGE_COLOR[1]} stroke={STAGE_STROKE[1]} label="② Reward Model" sub="偏好排序 → 标量打分器" highlight={activeStage === 1} />
      <Arrow x1={startX + 2 * stageW + gap} y1={y + stageH / 2} x2={startX + 2 * stageW + 2 * gap} y2={y + stageH / 2} faded={activeStage !== 1 && activeStage !== 2} />

      {/* Stage 3: PPO */}
      <Box x={startX + 2 * stageW + 2 * gap} y={y} w={stageW} h={stageH} fill={STAGE_COLOR[2]} stroke={STAGE_STROKE[2]} label="③ PPO + KL" sub="RM 当奖励 fine-tune" highlight={activeStage === 2} />

      {/* 底部数据流注脚 */}
      <text x={startX + stageW / 2} y={y + stageH + 32} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#6b7280">
        13K labeler 示范对
      </text>
      <text x={startX + stageW + gap + stageW / 2} y={y + stageH + 32} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#6b7280">
        33K 偏好排序对
      </text>
      <text x={startX + 2 * stageW + 2 * gap + stageW / 2} y={y + stageH + 32} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#6b7280">
        31K prompts × N samples
      </text>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        三阶段都用 \"人类反馈\" 注入对齐信号 · 缺一不可
      </text>
    </svg>
  );
}
