const W = 700;
const H = 320;

interface Props {
  trained: boolean;
}

export function RewardModelTrainingDiagram({ trained }: Props) {
  // 未训练:winner/loser 分数几乎重叠;训练后:winner 明显高于 loser
  const winnerScore = trained ? 2.3 : 0.15;
  const loserScore = trained ? -0.6 : -0.05;

  const scaleY = (v: number) => 170 - v * 26; // v in [-3,3] mapped to y

  const barX1 = 260;
  const barX2 = 420;
  const barW = 70;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Reward Model 训练示意图">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        64K 对偏好比较 → Bradley-Terry loss → RM 学会给 winner 打更高分
      </text>

      {/* pipeline: pairs -> RM -> scores */}
      <rect x={40} y={70} width={120} height={50} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.2} />
      <text x={100} y={90} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">(x, y_w, y_l)</text>
      <text x={100} y={104} textAnchor="middle" fontSize={8} fill="#78350f">64K 偏好对</text>

      <line x1={160} y1={95} x2={190} y2={95} stroke="var(--border)" strokeWidth={1.5} markerEnd="url(#arrow-rm)" />
      <defs>
        <marker id="arrow-rm" markerWidth={8} markerHeight={8} refX={4} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--border)" />
        </marker>
      </defs>

      <rect x={190} y={65} width={130} height={60} rx={6} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
      <text x={255} y={90} textAnchor="middle" fontSize={10} fontWeight={700} fill="#9d174d">Reward Model</text>
      <text x={255} y={104} textAnchor="middle" fontSize={8} fill="#9d174d">r_φ(x, y) → 标量</text>
      <text x={255} y={116} textAnchor="middle" fontSize={7} fill="#9d174d">初始化 = SFT 权重 + scalar head</text>

      {/* zero baseline */}
      <line x1={220} y1={170} x2={480} y2={170} stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="3,3" />
      <text x={490} y={174} fontSize={8} fill="var(--ink-muted)">0</text>

      {/* winner bar */}
      <rect
        x={barX1}
        y={winnerScore >= 0 ? scaleY(winnerScore) : 170}
        width={barW}
        height={Math.abs(scaleY(winnerScore) - 170)}
        fill="#ecfdf5"
        stroke="#10b981"
        strokeWidth={1.6}
      />
      <text x={barX1 + barW / 2} y={winnerScore >= 0 ? scaleY(winnerScore) - 8 : 184} textAnchor="middle" fontSize={10} fontWeight={700} fill="#065f46">
        r(y_w) = {winnerScore.toFixed(2)}
      </text>
      <text x={barX1 + barW / 2} y={264} textAnchor="middle" fontSize={9} fill="#374151">winner(被偏好)</text>

      {/* loser bar */}
      <rect
        x={barX2}
        y={loserScore >= 0 ? scaleY(loserScore) : 170}
        width={barW}
        height={Math.abs(scaleY(loserScore) - 170)}
        fill="#fef2f2"
        stroke="#ef4444"
        strokeWidth={1.6}
      />
      <text x={barX2 + barW / 2} y={loserScore >= 0 ? scaleY(loserScore) - 8 : 184} textAnchor="middle" fontSize={10} fontWeight={700} fill="#991b1b">
        r(y_l) = {loserScore.toFixed(2)}
      </text>
      <text x={barX2 + barW / 2} y={264} textAnchor="middle" fontSize={9} fill="#374151">loser(被拒绝)</text>

      <text x={W / 2} y={296} textAnchor="middle" fontSize={9} fontStyle="italic" fill="var(--ink-secondary)">
        L_RM = −log σ(r(y_w) − r(y_l)) — {trained ? "训练后差距拉开,loss 趋近 0" : "初始化时几乎重叠,loss 很大"}
      </text>
    </svg>
  );
}
