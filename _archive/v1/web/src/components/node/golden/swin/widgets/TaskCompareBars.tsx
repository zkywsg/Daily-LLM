import { TASK_COMPARE } from "../lib/data";

const W = 700;
const H = 280;

export function TaskCompareBars() {
  const PAD_L = 150;
  const PAD_R = 60;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Swin vs ResNet multi-task comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Swin-T vs ResNet — 分类 / 检测 / 分割全面领先
      </text>

      <g transform={`translate(${PAD_L}, 40)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">ResNet</text>
        <rect x={90} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={108} y={10} fontSize={10} fill="#374151">Swin</text>
      </g>

      {TASK_COMPARE.map((r, i) => {
        const y = PAD_T + i * (rowH + 8);
        const maxV = Math.max(r.resnet, r.swin) * 1.15;
        const wOf = (v: number) => (v / maxV) * plotW;
        return (
          <g key={r.task}>
            <text x={PAD_L - 8} y={y + 12} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{r.task}</text>
            <text x={PAD_L - 8} y={y + 26} textAnchor="end" fontSize={9} fill="#9ca3af">({r.metric})</text>

            <rect x={PAD_L} y={y} width={wOf(r.resnet)} height={20} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(r.resnet) + 6} y={y + 15} fontSize={9} fill="#6b7280">{r.resnet}</text>

            <rect x={PAD_L} y={y + 26} width={wOf(r.swin)} height={20} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(r.swin) + 6} y={y + 41} fontSize={9} fontWeight={700} fill="#ec4899">{r.swin}</text>

            <text x={W - 8} y={y + 20} textAnchor="end" fontSize={10} fontWeight={700} fill="#065f46">
              +{(r.swin - r.resnet).toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
