import { expertLoadAuxFree, expertLoadAuxLoss, NUM_ROUTED_SAMPLE } from "../lib/data";

interface Props {
  progress: number; // 0..1,训练进度(仅用于 aux-free 曲线)
  showAuxLoss: boolean; // 是否叠加显示传统 aux-loss 方案作对比
}

const W = 700;
const H = 340;

// Aux-loss-free:bias 随训练进度把负载拉平(不参与梯度,不干扰主 loss)。
// 对比传统 aux loss:靠额外 loss 梯度强制拉平,从头到尾都均衡但和主任务打架。

export function AuxLossFreeBalancingDiagram({ progress, showAuxLoss }: Props) {
  const auxFree = expertLoadAuxFree(progress);
  const auxLoss = expertLoadAuxLoss();
  const max = 3.5;

  const PAD = { left: 60, right: 30, top: 56, bottom: 56 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const groupW = innerW / NUM_ROUTED_SAMPLE;
  const barW = showAuxLoss ? groupW * 0.32 : groupW * 0.6;
  const yScale = (v: number) => PAD.top + (1 - v / max) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Aux-loss-free 负载均衡">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        Expert 负载(演示压缩到 8 个,实际 256 个)· 训练进度 {(progress * 100).toFixed(0)}%
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        紫色 = V3 aux-loss-free(bias 动态调整){showAuxLoss ? " · 灰色 = 传统 aux loss(Switch/Mixtral)" : ""}
      </text>

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} x2={W - PAD.right} y1={yScale(1)} y2={yScale(1)} stroke="#10b981" strokeWidth={1.4} strokeDasharray="4 3" />
      <text x={W - PAD.right - 4} y={yScale(1) - 5} textAnchor="end" fontSize={9} fontStyle="italic" fill="#10b981">
        理想均匀 = 1.0
      </text>

      {auxFree.map((v, i) => {
        const x = PAD.left + i * groupW + groupW / 2 - (showAuxLoss ? barW + 2 : barW / 2);
        const h = (v / max) * innerH;
        return (
          <g key={`af-${i}`}>
            <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} rx={2} fill="#6366f1" opacity={0.85} />
            <text x={x + barW / 2} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">
              E{i}
            </text>
          </g>
        );
      })}

      {showAuxLoss &&
        auxLoss.map((v, i) => {
          const x = PAD.left + i * groupW + groupW / 2 + 2;
          const h = (v / max) * innerH;
          return <rect key={`al-${i}`} x={x} y={H - PAD.bottom - h} width={barW} height={h} rx={2} fill="#9ca3af" opacity={0.85} />;
        })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {progress < 0.3
          ? "训练早期:bias 还没调好,负载仍不均衡"
          : progress < 0.7
            ? "训练中期:bias 持续修正,负载逐渐拉平"
            : "训练后期:bias 已收敛,负载接近均衡 —— 全程主 loss 不受干扰"}
      </text>
    </svg>
  );
}
