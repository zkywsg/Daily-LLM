import { PAGING_TIMELINE, GPU_CAPACITY_GB } from "../lib/data";

const W = 700;
const H = 280;

interface Props {
  showPaging: boolean;
}

export function PagedOptimizerChart({ showPaging }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 40;
  const PAD_B = 40;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const maxGB = 55;

  const xOf = (step: number) => PAD_L + (step / (PAGING_TIMELINE.length - 1)) * plotW;
  const yOf = (gb: number) => PAD_T + plotH - (gb / maxGB) * plotH;

  const path = PAGING_TIMELINE.map((p, i) => {
    const v = showPaging ? p.withPaging : p.withoutPaging;
    return `${i === 0 ? "M" : "L"} ${xOf(p.step)} ${yOf(v)}`;
  }).join(" ");

  const capacityY = yOf(GPU_CAPACITY_GB);
  const overflow = PAGING_TIMELINE.some((p) => p.withoutPaging > GPU_CAPACITY_GB);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="paged optimizer 显存占用演示">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {showPaging ? "Paged Optimizer — 状态分页到 CPU,显存曲线被削平" : "无 Paging — 显存 spike 可能超出 GPU 容量(OOM)"}
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <text x={PAD_L + plotW / 2} y={H - 10} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">训练 step</text>

      <line x1={PAD_L} y1={capacityY} x2={PAD_L + plotW} y2={capacityY} stroke="#ec4899" strokeWidth={1.4} strokeDasharray="4 2" />
      <text x={PAD_L + plotW} y={capacityY - 6} textAnchor="end" fontSize={9} fill="#ec4899">GPU 容量 {GPU_CAPACITY_GB}GB</text>

      <path d={path} fill="none" stroke={showPaging ? "#10b981" : "#3b82f6"} strokeWidth={2.4} />
      {PAGING_TIMELINE.map((p, i) => {
        const v = showPaging ? p.withPaging : p.withoutPaging;
        const isOverflow = !showPaging && v > GPU_CAPACITY_GB;
        return (
          <circle key={i} cx={xOf(p.step)} cy={yOf(v)} r={isOverflow ? 6 : 4} fill={isOverflow ? "#ef4444" : showPaging ? "#10b981" : "#3b82f6"} />
        );
      })}

      {!showPaging && overflow && (
        <text x={W / 2} y={PAD_T - 8} textAnchor="middle" fontSize={10} fontWeight={700} fill="#ef4444">
          红点 = 显存需求超出 GPU 容量,训练会 OOM
        </text>
      )}
    </svg>
  );
}
