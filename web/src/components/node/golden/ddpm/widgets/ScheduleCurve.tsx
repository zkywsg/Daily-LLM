import { useMemo } from "react";
import {
  alphaBarCumulative,
  alphaFromBeta,
  betaSchedule,
  snr,
  type Schedule,
} from "../lib/math";

interface Props {
  T: number;
  t: number;
  schedule: Schedule;
  /** 画哪些曲线;默认全画 */
  show?: { beta?: boolean; alphaBar?: boolean; snr?: boolean };
}

const W = 700;
const H = 240;
const PAD = { left: 50, right: 60, top: 28, bottom: 32 };

// 三条曲线:β_t (噪声步长)、ᾱ_t (累积保留量)、log10 SNR(t)。
// 当前 t 用一条粉色竖线标出。
// 让 viewer 直观看到:为啥 linear 在尾部 SNR 已经很低、cosine 则平滑得多。
export function ScheduleCurve({ T, t, schedule, show = {} }: Props) {
  const { beta, aBar, snrLog } = useMemo(() => {
    const b = betaSchedule(T, schedule);
    const a = alphaFromBeta(b);
    const ab = alphaBarCumulative(a);
    const sn = ab.map((x) => Math.log10(Math.max(1e-8, snr(x))));
    return { beta: b, aBar: ab, snrLog: sn };
  }, [T, schedule]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const xScale = (i: number) => PAD.left + (i / Math.max(1, T - 1)) * innerW;

  // 左轴:β & ᾱ ∈ [0, 1]
  const yLeft = (v: number) => PAD.top + (1 - v) * innerH;
  // 右轴:log10 SNR ∈ [-3, 4]
  const SNR_MIN = -3;
  const SNR_MAX = 4;
  const yRight = (v: number) => PAD.top + (1 - (v - SNR_MIN) / (SNR_MAX - SNR_MIN)) * innerH;

  const polyline = (vals: number[], yFn: (v: number) => number) =>
    vals.map((v, i) => `${xScale(i)},${yFn(v)}`).join(" ");

  const showBeta = show.beta !== false;
  const showAB = show.alphaBar !== false;
  const showSNR = show.snr !== false;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="DDPM β / ᾱ / SNR schedule curves"
    >
      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={W - PAD.right} y1={PAD.top} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* 左轴刻度 */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={`l-${v}`}>
          <text x={PAD.left - 5} y={yLeft(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {v.toFixed(2)}
          </text>
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={yLeft(v)}
            y2={yLeft(v)}
            stroke="var(--border)"
            strokeDasharray="1 4"
          />
        </g>
      ))}

      {/* 右轴刻度 (log10 SNR) */}
      {[-3, -1, 1, 3].map((v) => (
        <text key={`r-${v}`} x={W - PAD.right + 5} y={yRight(v) + 4} fontSize={10} fill="#3b82f6">
          {v >= 0 ? `1e${v}` : `1e${v}`}
        </text>
      ))}

      {/* x 轴 */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        diffusion step t →
      </text>

      {/* 当前 t 竖线 */}
      <line
        x1={xScale(t)}
        x2={xScale(t)}
        y1={PAD.top}
        y2={H - PAD.bottom}
        stroke="#ec4899"
        strokeWidth={1.5}
        strokeDasharray="3 3"
      />

      {/* 曲线 */}
      {showBeta && (
        <polyline fill="none" stroke="#f59e0b" strokeWidth={2} points={polyline(beta, yLeft)} />
      )}
      {showAB && (
        <polyline fill="none" stroke="#10b981" strokeWidth={2} points={polyline(aBar, yLeft)} />
      )}
      {showSNR && (
        <polyline fill="none" stroke="#3b82f6" strokeWidth={2} points={polyline(snrLog, yRight)} />
      )}

      {/* 图例 */}
      <g transform={`translate(${PAD.left + 10}, ${PAD.top + 6})`}>
        {showBeta && (
          <g transform="translate(0, 0)">
            <rect width={12} height={3} fill="#f59e0b" />
            <text x={16} y={4} fontSize={10} fill="var(--ink-secondary)">β_t (噪声步长)</text>
          </g>
        )}
        {showAB && (
          <g transform="translate(0, 14)">
            <rect width={12} height={3} fill="#10b981" />
            <text x={16} y={4} fontSize={10} fill="var(--ink-secondary)">ᾱ_t (信号保留比)</text>
          </g>
        )}
        {showSNR && (
          <g transform="translate(0, 28)">
            <rect width={12} height={3} fill="#3b82f6" />
            <text x={16} y={4} fontSize={10} fill="var(--ink-secondary)">SNR(t) (log10, 右轴)</text>
          </g>
        )}
      </g>
    </svg>
  );
}
