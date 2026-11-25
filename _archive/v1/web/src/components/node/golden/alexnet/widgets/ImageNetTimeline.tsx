import { IMAGENET_TIMELINE } from "../lib/data";

const W = 700;
const H = 320;

// 2010-2015 ImageNet Top-5 错误率历年曲线,AlexNet 2012 标红
export function ImageNetTimeline() {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const years = IMAGENET_TIMELINE;
  const yMin = 0, yMax = 30;
  const xOf = (i: number) => PAD_L + (i / (years.length - 1)) * plotW;
  const yOf = (e: number) => PAD_T + ((yMax - e) / (yMax - yMin)) * plotH;

  const pts = years.map((y, i) => `${xOf(i)},${yOf(y.topFive)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ImageNet ILSVRC top-5 error rate 2010-2015">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        ImageNet ILSVRC Top-5 错误率 — 2012 是 CNN 时代起点
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 10, 20, 30].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}%</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {/* human baseline */}
      <line x1={PAD_L} y1={yOf(5.1)} x2={W - PAD_R} y2={yOf(5.1)} stroke="#10b981" strokeWidth={1.5} strokeDasharray="4 4" />
      <text x={W - PAD_R - 4} y={yOf(5.1) - 6} textAnchor="end" fontSize={10} fontWeight={600} fill="#065f46">human baseline 5.1%</text>

      {/* AlexNet split (between 2011 and 2012) */}
      <line x1={(xOf(1) + xOf(2)) / 2} y1={PAD_T} x2={(xOf(1) + xOf(2)) / 2} y2={PAD_T + plotH} stroke="#ec4899" strokeWidth={1.5} strokeDasharray="3 3" />
      <text x={(xOf(1) + xOf(2)) / 2 + 6} y={PAD_T + 14} fontSize={10} fontWeight={700} fill="#831843">CNN era →</text>

      {/* curve */}
      <polyline points={pts} fill="none" stroke="#3b82f6" strokeWidth={2.5} />

      {years.map((y, i) => {
        const isAlex = y.method.startsWith("AlexNet");
        return (
          <g key={y.year}>
            <circle cx={xOf(i)} cy={yOf(y.topFive)} r={isAlex ? 7 : 5} fill={isAlex ? "#ec4899" : y.isCNN ? "#3b82f6" : "#9ca3af"} stroke="#fff" strokeWidth={2} />
            <text x={xOf(i)} y={yOf(y.topFive) - 12} textAnchor="middle" fontSize={10} fontWeight={isAlex ? 700 : 500} fill={isAlex ? "#831843" : "#374151"}>
              {y.topFive}%
            </text>
            <text x={xOf(i)} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={10} fill="#374151">{y.year}</text>
            <text x={xOf(i)} y={PAD_T + plotH + 30} textAnchor="middle" fontSize={9} fill="#6b7280">{y.method.split(" ")[0]}</text>
          </g>
        );
      })}
    </svg>
  );
}
