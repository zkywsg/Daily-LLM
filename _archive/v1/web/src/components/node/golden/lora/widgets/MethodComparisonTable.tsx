import { METHOD_COMPARE } from "../lib/math";

// PEFT 方法对比表:Full / Adapter / Prefix / LoRA。
// 高亮 LoRA 行 —— 它是参数比 Adapter 小、推理延迟为 0、GLUE 几乎不掉的唯一组合。

const W = 720;
const H = 240;

export function MethodComparisonTable() {
  const rowH = 40;
  const headerH = 30;
  const cols = [
    { x: 14, w: 170, key: "label", label: "方法" },
    { x: 184, w: 110, key: "paramPct", label: "trainable %" },
    { x: 294, w: 130, key: "latency", label: "推理延迟" },
    { x: 424, w: 90, key: "glue", label: "ΔGLUE" },
    { x: 514, w: 200, key: "note", label: "特征" },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PEFT 方法对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Full vs Adapter vs Prefix vs LoRA(数量级取自 LoRA 论文 RoBERTa-large)
      </text>

      {/* header */}
      <rect x={0} y={30} width={W} height={headerH} fill="var(--bg-subtle)" />
      {cols.map((c) => (
        <text key={c.key} x={c.x} y={50} fontSize={11} fontWeight={600} fill="var(--ink-secondary)">
          {c.label}
        </text>
      ))}

      {/* rows */}
      {METHOD_COMPARE.map((m, i) => {
        const y = 30 + headerH + i * rowH;
        const highlight = m.label.startsWith("LoRA");
        return (
          <g key={m.label}>
            {highlight && <rect x={0} y={y} width={W} height={rowH} fill="#dbeafe" opacity={0.35} />}
            <line x1={0} y1={y} x2={W} y2={y} stroke="var(--border)" />
            <circle cx={cols[0].x + 6} cy={y + rowH / 2} r={5} fill={m.color} />
            <text x={cols[0].x + 18} y={y + rowH / 2 + 4} fontSize={11} fontWeight={highlight ? 700 : 500} fill="var(--ink-primary)">
              {m.label}
            </text>
            <text x={cols[1].x} y={y + rowH / 2 + 4} fontSize={11} fill="var(--ink-secondary)">
              {m.paramPct < 1 ? `${m.paramPct.toFixed(1)}%` : `${m.paramPct}%`}
            </text>
            <text x={cols[2].x} y={y + rowH / 2 + 4} fontSize={11} fill={m.inferenceLatency ? "#dc2626" : "#10b981"} fontWeight={600}>
              {m.inferenceLatency ? "+ 额外延迟" : "✓ 零延迟"}
            </text>
            <text x={cols[3].x} y={y + rowH / 2 + 4} fontSize={11} fill={m.glueDelta >= -0.2 ? "#10b981" : "#dc2626"}>
              {m.glueDelta > 0 ? "+" : ""}
              {m.glueDelta.toFixed(1)}
            </text>
            <text x={cols[4].x} y={y + rowH / 2 + 4} fontSize={10} fill="var(--ink-muted)">
              {m.note}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
