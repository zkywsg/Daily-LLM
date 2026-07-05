import { TRAINING_STEP_GROWTH } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  visibleSteps: number;
}

export function ThinkingGrowthChart({ visibleSteps }: Props) {
  const points = TRAINING_STEP_GROWTH.slice(0, visibleSteps);
  const PAD_L = 60;
  const PAD_R = 50;
  const PAD_T = 40;
  const PAD_B = 40;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const maxStep = 8000;
  const maxTokens = 10000;
  const maxAime = 80;

  const xOf = (step: number) => PAD_L + (step / maxStep) * plotW;
  const yTokOf = (tok: number) => PAD_T + plotH - (tok / maxTokens) * plotH;
  const yAimeOf = (acc: number) => PAD_T + plotH - (acc / maxAime) * plotH;

  const tokenPath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p.step)} ${yTokOf(p.thinkTokens)}`).join(" ");
  const aimePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p.step)} ${yAimeOf(p.aime)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="R1-Zero 训练中思考长度与 AIME 准确率联合增长">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        R1-Zero 训练曲线 — 没有人为加 length reward,思考长度自然增长
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <text x={PAD_L + plotW / 2} y={H - 8} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">训练 step</text>

      <path d={tokenPath} fill="none" stroke="#3b82f6" strokeWidth={2.4} />
      <path d={aimePath} fill="none" stroke="#10b981" strokeWidth={2.4} strokeDasharray="4 2" />

      {points.map((p, i) => (
        <g key={i}>
          <circle cx={xOf(p.step)} cy={yTokOf(p.thinkTokens)} r={4} fill="#3b82f6" />
          <circle cx={xOf(p.step)} cy={yAimeOf(p.aime)} r={4} fill="#10b981" />
          {p.hasAha && (
            <g>
              <text x={xOf(p.step)} y={yTokOf(p.thinkTokens) - 12} textAnchor="middle" fontSize={14}>💡</text>
            </g>
          )}
        </g>
      ))}

      <g transform={`translate(${PAD_L}, ${PAD_T - 20})`}>
        <line x1={0} y1={-4} x2={16} y2={-4} stroke="#3b82f6" strokeWidth={2.4} />
        <text x={20} y={0} fontSize={9} fill="#3b82f6">thinking tokens</text>
        <line x1={140} y1={-4} x2={156} y2={-4} stroke="#10b981" strokeWidth={2.4} strokeDasharray="4 2" />
        <text x={160} y={0} fontSize={9} fill="#10b981">AIME 准确率</text>
      </g>

      {points.some((p) => p.hasAha) && (
        <text x={W - PAD_R} y={PAD_T + 10} textAnchor="end" fontSize={9} fill="#b45309">💡 = "Aha moment"(自发反思语言涌现)</text>
      )}
    </svg>
  );
}
