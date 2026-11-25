import { TEST_TIME_SCALING } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  activeTask: string;
}

const TASK_COLORS: Record<string, string> = {
  AIME: "#3b82f6",
  Codeforces: "#ec4899",
  GPQA: "#10b981",
};

export function TestTimeScalingChart({ activeTask }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 40;
  const PAD_B = 40;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const logMin = Math.log10(300);
  const logMax = Math.log10(80000);
  const xOf = (t: number) => PAD_L + ((Math.log10(t) - logMin) / (logMax - logMin)) * plotW;
  const yOf = (acc: number) => PAD_T + plotH - (acc / 100) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="test-time compute scaling 曲线">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Test-Time Compute Scaling — thinking tokens 翻倍,准确率持续涨
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <text x={PAD_L - 10} y={PAD_T + 4} textAnchor="end" fontSize={9} fill="var(--ink-muted)">100%</text>
      <text x={PAD_L - 10} y={PAD_T + plotH + 4} textAnchor="end" fontSize={9} fill="var(--ink-muted)">0%</text>
      <text x={PAD_L} y={PAD_T + plotH + 20} fontSize={9} fill="var(--ink-muted)">~300 tok</text>
      <text x={PAD_L + plotW - 20} y={PAD_T + plotH + 20} fontSize={9} fill="var(--ink-muted)">~80K tok</text>
      <text x={PAD_L + plotW / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">thinking tokens(log 尺度)</text>

      {Object.entries(TEST_TIME_SCALING).map(([task, points], taskIdx) => {
        const isFocus = task === activeTask || activeTask === "all";
        const color = TASK_COLORS[task];
        const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p.thinkTokens)} ${yOf(p.accuracy)}`).join(" ");
        const labelYOffset = (taskIdx - 1) * 12;
        return (
          <g key={task} opacity={isFocus ? 1 : 0.15}>
            <path d={path} fill="none" stroke={color} strokeWidth={2.4} />
            {points.map((p, i) => (
              <circle key={i} cx={xOf(p.thinkTokens)} cy={yOf(p.accuracy)} r={4} fill={color} />
            ))}
            <text x={xOf(points[points.length - 1].thinkTokens) + 8} y={yOf(points[points.length - 1].accuracy) + 4 + labelYOffset} fontSize={10} fontWeight={700} fill={color}>
              {task}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
