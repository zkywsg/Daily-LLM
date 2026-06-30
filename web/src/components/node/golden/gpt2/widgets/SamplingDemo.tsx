import { applyTemperature, applyTopK, applyTopP } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  strategy: "greedy" | "temp" | "topk" | "topp";
  temperature: number;
  topK: number;
  topP: number;
}

// 模拟一个 LM next-token 分布:对 8 个候选词的 logits
const CANDIDATES = ["cat", "dog", "bird", "fox", "wolf", "tiger", "lion", "horse"];
const BASE_LOGITS = [3.2, 2.8, 2.0, 1.6, 1.1, 0.7, 0.4, 0.1];

export function SamplingDemo({ strategy, temperature, topK, topP }: Props) {
  let probs: number[];
  if (strategy === "greedy") {
    probs = BASE_LOGITS.map((_, i) => i === 0 ? 1 : 0);
  } else {
    const tempProbs = applyTemperature(BASE_LOGITS, strategy === "temp" ? temperature : 1.0);
    if (strategy === "topk") probs = applyTopK(tempProbs, topK);
    else if (strategy === "topp") probs = applyTopP(tempProbs, topP);
    else probs = tempProbs;
  }

  const PAD_L = 80;
  const PAD_R = 30;
  const PAD_T = 60;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const barW = (plotW / CANDIDATES.length) - 8;
  const yOf = (p: number) => PAD_T + (1 - p) * plotH;

  const strategyDesc =
    strategy === "greedy" ? "确定:只选 argmax,无多样性"
    : strategy === "temp" ? `平滑 logits·temp=${temperature.toFixed(2)} → ${temperature < 1 ? "更尖" : temperature > 1 ? "更平" : "原始"}`
    : strategy === "topk" ? `保留 top-${topK},其他归零再归一化`
    : `累积概率 ≥ ${topP.toFixed(2)} 即截止,适应分布的 nucleus`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LM sampling strategies demo">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        下一个 token 的采样分布 — {strategy.toUpperCase()}
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {strategyDesc}
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />
      {[0, 0.25, 0.5, 0.75, 1.0].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y.toFixed(2)}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {CANDIDATES.map((tok, i) => {
        const x = PAD_L + 8 + i * (barW + 8);
        const p = probs[i];
        const h = (PAD_T + plotH) - yOf(p);
        const isZero = p < 0.001;
        return (
          <g key={i}>
            <rect x={x} y={yOf(p)} width={barW} height={h} fill={isZero ? "#f3f4f6" : "#fce7f3"} stroke={isZero ? "#d1d5db" : "#ec4899"} strokeWidth={1.2} rx={2} />
            <text x={x + barW / 2} y={yOf(p) - 4} textAnchor="middle" fontSize={9} fontWeight={isZero ? 400 : 600} fill={isZero ? "#9ca3af" : "#831843"}>
              {(p * 100).toFixed(0)}%
            </text>
            <text x={x + barW / 2} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={10} fill="#374151">{tok}</text>
            <text x={x + barW / 2} y={PAD_T + plotH + 28} textAnchor="middle" fontSize={8} fill="#9ca3af">{BASE_LOGITS[i].toFixed(1)}</text>
          </g>
        );
      })}
      <text x={PAD_L - 8} y={PAD_T + plotH + 28} textAnchor="end" fontSize={9} fill="#9ca3af">logits</text>
    </svg>
  );
}
