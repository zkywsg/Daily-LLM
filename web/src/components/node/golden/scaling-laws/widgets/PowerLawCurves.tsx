import { ALPHA_N, ALPHA_D, ALPHA_C, lossN, lossD, lossC } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  axis: "N" | "D" | "C" | "all";
}

const SERIES = [
  { key: "N", name: "L(N) · 参数",  color: "#ec4899", alpha: ALPHA_N, xMin: 1e7,  xMax: 1e12, fn: lossN },
  { key: "D", name: "L(D) · 数据",  color: "#3b82f6", alpha: ALPHA_D, xMin: 1e7,  xMax: 1e12, fn: lossD },
  { key: "C", name: "L(C) · 算力",  color: "#10b981", alpha: ALPHA_C, xMin: 1e2,  xMax: 1e9,  fn: lossC },
];

export function PowerLawCurves({ axis }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // 用归一化:每条曲线 x 取 log,自己范围内 [0, 1]
  // y log,[0.5, 5] 这种宽 loss 范围
  const yMin = 1.5, yMax = 7;
  const yLogMin = Math.log10(yMin), yLogMax = Math.log10(yMax);
  const yOf = (l: number) => PAD_T + ((yLogMax - Math.log10(l)) / (yLogMax - yLogMin)) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Three power law curves">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        三轴幂律 — log-log 直线 · L ∝ x^(-α)
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* y ticks */}
      {[2, 3, 4, 5, 6].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y.toFixed(1)}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {/* x ticks (归一化 0..1 表 10 个数量级) */}
      {[0, 0.25, 0.5, 0.75, 1].map((u, i) => (
        <g key={i}>
          <line x1={PAD_L + u * plotW} y1={PAD_T + plotH} x2={PAD_L + u * plotW} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={PAD_L + u * plotW} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">+{i}</text>
        </g>
      ))}

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">log10(x / x_min) · 5 个数量级</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">test loss (log)</text>

      {SERIES.map((s) => {
        const isVisible = axis === "all" || axis === s.key;
        const dim = isVisible ? 1 : 0.18;
        const pts: string[] = [];
        for (let i = 0; i <= 200; i++) {
          const u = i / 200;
          const logX = Math.log10(s.xMin) + u * (Math.log10(s.xMax) - Math.log10(s.xMin));
          const x = Math.pow(10, logX);
          const l = s.fn(x);
          if (l >= yMin && l <= yMax) {
            pts.push(`${PAD_L + u * plotW},${yOf(l)}`);
          }
        }
        return (
          <g key={s.key} opacity={dim}>
            <polyline points={pts.join(" ")} fill="none" stroke={s.color} strokeWidth={2.5} />
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        {SERIES.map((s, i) => (
          <g key={s.key} transform={`translate(0, ${i * 18})`} opacity={axis === "all" || axis === s.key ? 1 : 0.4}>
            <line x1={0} y1={6} x2={20} y2={6} stroke={s.color} strokeWidth={2.5} />
            <text x={26} y={10} fontSize={11} fontWeight={600} fill={s.color}>{s.name}</text>
            <text x={130} y={10} fontSize={10} fill="#6b7280">α = {s.alpha.toFixed(3)}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
