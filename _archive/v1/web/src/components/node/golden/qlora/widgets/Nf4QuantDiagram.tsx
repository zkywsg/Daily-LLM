import { NF4_LEVELS, INT4_LEVELS, normalPdf } from "../lib/data";

const W = 700;
const H = 280;

interface Props {
  mode: "nf4" | "int4";
}

export function Nf4QuantDiagram({ mode }: Props) {
  const PAD_L = 40;
  const PAD_R = 40;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const curveBaseY = 160;
  const curveH = 100;

  const xOf = (v: number) => PAD_L + ((v + 1.2) / 2.4) * plotW;

  const levels = mode === "nf4" ? NF4_LEVELS : INT4_LEVELS;

  const curvePoints: string[] = [];
  for (let i = -1.2; i <= 1.2; i += 0.02) {
    const x = xOf(i);
    const y = curveBaseY - normalPdf(i) * curveH * 0.9;
    curvePoints.push(`${curvePoints.length === 0 ? "M" : "L"} ${x} ${y}`);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="NF4 vs INT4 量化位点分布对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "nf4" ? "NF4 — 量化位点取自正态分布等分位点" : "INT4 — 量化位点均匀分布(次优)"}
      </text>

      <path d={curvePoints.join(" ")} fill="none" stroke="#9ca3af" strokeWidth={1.6} strokeDasharray="4 2" />
      <text x={W - PAD_R} y={curveBaseY - curveH - 4} textAnchor="end" fontSize={9} fill="#9ca3af">权重的真实分布(正态)</text>

      {levels.map((v, i) => {
        const x = xOf(v);
        const density = normalPdf(v);
        const isDense = density > normalPdf(0) * 0.5;
        const color = mode === "nf4" ? "#10b981" : "#ec4899";
        return (
          <g key={i}>
            <line x1={x} y1={curveBaseY} x2={x} y2={curveBaseY + 40} stroke={color} strokeWidth={isDense ? 2 : 1.2} />
            <circle cx={x} cy={curveBaseY + 40} r={3} fill={color} />
          </g>
        );
      })}

      <text x={W / 2} y={curveBaseY + 65} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {mode === "nf4"
          ? "位点在 0 附近(权重密集区)更密,远端(权重稀疏区)更疏 — 整体量化误差比 INT4 小 ~30%"
          : "位点等距分布,0 附近(权重最密集处)反而位点稀疏 — 大量权重挤在少数几个量化级别里,精度损失大"}
      </text>
    </svg>
  );
}
