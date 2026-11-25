import { useMemo } from "react";
import {
  alphaBarCumulative,
  alphaFromBeta,
  betaSchedule,
  forwardSample,
  makeDemoSignal,
  seededGaussian,
  type Schedule,
} from "../lib/math";

interface Props {
  T: number;
  t: number;
  schedule: Schedule;
  /** 像素数(信号长度);默认 64 */
  n?: number;
  /** 是否同时画"干净 x_0 baseline"虚线对照 */
  showBaseline?: boolean;
}

const W = 700;
const H = 220;
const PAD = { left: 36, right: 16, top: 30, bottom: 30 };

// 把一个 1D 信号当 "横向 image",每个像素一个色块(深→浅映射 [-,+] 值域),
// 信号曲线叠在上面。t 拖大时颜色变 noisy。
// 比直接画 2D 图像更轻、信息密度更高,还能配合曲线看到 SNR 衰减。
export function NoiseStrip({
  T,
  t,
  schedule,
  n = 64,
  showBaseline = true,
}: Props) {
  const { x0, xt, alphaBar_t } = useMemo(() => {
    const beta = betaSchedule(T, schedule);
    const alpha = alphaFromBeta(beta);
    const aBar = alphaBarCumulative(alpha);
    const x0 = makeDemoSignal(n);
    const eps = seededGaussian(n, 42);
    const at = aBar[Math.max(0, Math.min(T - 1, t))];
    const xt = forwardSample(x0, at, eps);
    return { x0, xt, alphaBar_t: at };
  }, [T, t, schedule, n]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const cellW = innerW / n;

  // 像素色块:把 xt 值钳到 [-3,3] 再映射到色阶
  const cellFill = (v: number): string => {
    const c = Math.max(-3, Math.min(3, v));
    const norm = (c + 3) / 6; // 0..1
    // 用粉->蓝调色板:低 = 蓝 (cold,负值),高 = 粉 (hot,正值)
    const hue = norm > 0.5 ? 330 : 220;
    const light = 95 - Math.abs(c) * 15;
    return `hsl(${hue}, 70%, ${Math.max(35, light)}%)`;
  };

  const yScale = (v: number) => PAD.top + ((1 - (v + 3) / 6) * innerH);
  const baselinePts = x0.map((v, i) => `${PAD.left + (i + 0.5) * cellW},${yScale(v)}`).join(" ");
  const noisyPts = xt.map((v, i) => `${PAD.left + (i + 0.5) * cellW},${yScale(v)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Forward sample at t=${t}, ᾱ_t=${alphaBar_t.toFixed(3)}`}
    >
      {/* 像素色带 */}
      {xt.map((v, i) => (
        <rect
          key={i}
          x={PAD.left + i * cellW}
          y={PAD.top}
          width={cellW + 0.5}
          height={innerH}
          fill={cellFill(v)}
          opacity={0.55}
        />
      ))}

      {/* baseline 虚线 = 干净 x_0 */}
      {showBaseline && (
        <polyline
          fill="none"
          stroke="#9ca3af"
          strokeWidth={1}
          strokeDasharray="3 3"
          points={baselinePts}
        />
      )}

      {/* 当前 x_t 曲线 */}
      <polyline
        fill="none"
        stroke="#ec4899"
        strokeWidth={1.8}
        points={noisyPts}
      />

      {/* 标签 */}
      <text x={PAD.left} y={18} fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        x_t (t = {t} / {T - 1})
      </text>
      <text x={W - PAD.right} y={18} textAnchor="end" fontSize={11} fill="var(--ink-muted)">
        ᾱ_t = {alphaBar_t.toFixed(3)} · √(ᾱ_t) signal + √(1-ᾱ_t) noise
      </text>
      <text x={W - PAD.right} y={H - 8} textAnchor="end" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        粉=正值 · 蓝=负值 · 虚线=原始 x_0
      </text>
    </svg>
  );
}
