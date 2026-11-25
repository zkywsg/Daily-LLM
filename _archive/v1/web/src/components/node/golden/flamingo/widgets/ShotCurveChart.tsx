import { SHOT_CURVE } from "../lib/data";

const W = 700;
const H = 300;

export function ShotCurveChart() {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xVals = [0, 4, 32];
  const xPos = (s: number) => PAD_L + (xVals.indexOf(s) / (xVals.length - 1)) * plotW;

  const yMin = 25, yMax = 65;
  const yOf = (v: number) => PAD_T + ((yMax - v) / (yMax - yMin)) * plotH;

  const series = [
    { key: "vqav2" as const, name: "VQAv2", color: "#ec4899" },
    { key: "okvqa" as const, name: "OK-VQA", color: "#3b82f6" },
    { key: "textvqa" as const, name: "TextVQA", color: "#10b981" },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Shot count vs performance curve">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        In-Context Learning 曲线 — 0 → 4 → 32 shot
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[30, 40, 50, 60].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y}</text>
        </g>
      ))}
      {xVals.map((s) => (
        <g key={s}>
          <line x1={xPos(s)} y1={PAD_T + plotH} x2={xPos(s)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xPos(s)} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">{s}-shot</text>
        </g>
      ))}

      {series.map((s) => {
        const pts = SHOT_CURVE.map((r) => `${xPos(r.shots)},${yOf(r[s.key])}`).join(" ");
        return (
          <g key={s.key}>
            <polyline points={pts} fill="none" stroke={s.color} strokeWidth={2.4} />
            {SHOT_CURVE.map((r) => (
              <circle key={r.shots} cx={xPos(r.shots)} cy={yOf(r[s.key])} r={5} fill={s.color} stroke="#fff" strokeWidth={1.5} />
            ))}
          </g>
        );
      })}

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        {series.map((s, i) => (
          <g key={s.key} transform={`translate(${i * 130}, 0)`}>
            <line x1={0} y1={6} x2={20} y2={6} stroke={s.color} strokeWidth={2.5} />
            <text x={26} y={10} fontSize={10} fontWeight={600} fill={s.color}>{s.name}</text>
          </g>
        ))}
      </g>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        0→32-shot 提升 5-11 分 · 完全没在这种任务上训过,只靠 prompt 里的几个例子学会格式
      </text>
    </svg>
  );
}
